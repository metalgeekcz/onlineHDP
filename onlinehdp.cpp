/*
 * Copyright (C) 2013 by
 * 
 * Cheng Zhang
 * chengz@kth.se
 * and
 * Xavi Gratal
 * javiergm@kth.se
 * Computer Vision and Active Perception Lab
 * KTH Royal Institue of Techonology
 *
 * onlineHDP_C++ is a free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published
 * by the Free Software Foundation; either version 2 of the License,
 * or (at your option) any later version.
 *
 * onlineHDP_C++ is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with onlineHDP_c++; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA.
 */
#include "onlinehdp.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "utils.h"
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/random.hpp>
#include <boost/random/gamma_distribution.hpp>

using namespace Eigen;
using std::abs;

namespace HDP {

void model::init_hdp(){
    //parameters have been parsed

    // initial the sticks
    m_var_sticks = MatrixXd::Ones(2, T-1);
    //set the second row from T-1 to 1 in a descend orer
    for(int ii=0; ii<T-1; ii++){
        m_var_sticks(1,ii)=T-1-ii;
    }
    
    m_varphi_ss = VectorXd::Zero(T);

    //inital lambda size
    m_lambda = MatrixXd::Zero(T,W);
    //generate gamma random
    boost::mt19937 rng;
    boost::gamma_distribution<> d_gamma(1.0);
    boost::variate_generator<boost::mt19937&, boost::gamma_distribution<> > generator(rng, d_gamma); 
    
    for(int ii=0; ii<T*W; ii++){
        m_lambda(ii/W, ii%W) = generator();
    }
 
    m_lambda = m_lambda *D *100 /(T*W) - MatrixXd::Constant(T,W,eta);
    m_lambda_sum = m_lambda. rowwise().sum();
    m_eta = eta;
    //Eq 6 in online LDA paper E[logbeta_kw] = psi(lambda_kw) -psi(sum^W lambda_kw)
    m_Elogbeta = dirichlet_expectation(MatrixXd::Constant(T,W,m_eta)+m_lambda);
    

    m_tau = tau +1;
    m_adding_noise = adding_noise;
    m_T=T;
    m_alpha  = alpha;
    min_adding_noise_point = 10;
    int min_adding_noise_radio =  1;
    m_updatect = 0;
    m_status_up_to_date = true;
    m_num_docs_parsed = 0;

    //time stamps and normalizers for lazy updates
    m_timestamp = VectorXd::Zero(W);
    m_r.push_back(0);
    assert(m_r.size() == 1);

    //same para
    rhot_bound = 0.0;

}

 MatrixXd model::dirichlet_expectation(const MatrixXd & alpha){
     // For a vector theta ~ Dir(alpha), compute E[log(theta)] given alpha
     MatrixXd DirExp(alpha.rows(), alpha.cols());
     MatrixXd dig_alpha(alpha.rows(), alpha.cols());
     for(int rr=0; rr < alpha.rows(); rr++){
         for ( int cc=0; cc< alpha.cols(); cc++){
             dig_alpha(rr,cc) = boost::math::digamma(alpha(rr,cc));
         }
     }
     if (alpha.cols() == 1) {
         // it is a vector in this case
         DirExp=dig_alpha.array()-boost::math::digamma(alpha.sum());

     }else if (alpha.cols() > 1){
         // it is a matrix in this case
         MatrixXd sum_1 = alpha.rowwise().sum();
         MatrixXd dig_sum1(alpha.rows(), 1);
         for(int rr=0; rr < alpha.rows(); rr++){
             dig_sum1(rr,0) = boost::math::digamma(sum_1(rr,0));
         }
         const int cc=alpha.cols();
         DirExp= dig_alpha - dig_sum1.replicate(1,alpha.cols());
     }

     return DirExp;

 }

 VectorXd model::expect_log_sticks(const MatrixXd & sticks){
     //For stick-breaking hdp, this returns the E[log(sticks)]
     // it is initaled as 2 * T-1 Matrix. The first row is 1, the seocnd row is T-1:-1:1
     VectorXd ssum=sticks.colwise().sum();
     int sticks_len=sticks.cols();
     VectorXd dig_sum(sticks_len);
     VectorXd dig_para1(sticks_len);
     VectorXd dig_para2(sticks_len);
     for(int ii=0; ii < sticks.cols(); ++ii){
         dig_sum(ii) = boost::math::digamma(ssum(ii));
         dig_para1(ii) = boost::math::digamma(sticks(0,ii));
         dig_para2(ii) = boost::math::digamma(sticks(1,ii));
     }
     //ElogW is E[log beta']
     VectorXd ElogW = dig_para1 - dig_sum;
     //Elog!_W is E[log(1-beta')] 
     VectorXd Elog1_W = dig_para2 -dig_sum;

     int n = sticks_len + 1;
     VectorXd Elogsticks = VectorXd::Zero(n);
     Elogsticks.head(n-1) = ElogW;
     //compute E[log(beta/pi)] = E[log beta'] + sum^{k-1}[log( 1 - beta' )]
     for(int ii=1; ii<n; ii++ ){
         Elogsticks(ii) += Elog1_W.head(ii).sum();
     }

     return Elogsticks;
    
 }

double model::doc_e_step( document & doc,suff_stats & ss,
                        const  VectorXd & Elogsticks, std::vector<int> & word_list,
                        std::unordered_map<int, int> & unique_words){
    //e step for a single  document
    std::vector<int> batchids;
    // batchids are the value in unique_words correspond to every word of the document
    // the value in the unique_words record the order of the words has been seen
    //eg. training corpus [2 8 6 2] unique_words will be 2:0 8:1 6:2
    for( int ii=0; ii< doc.words.size(); ii++){
        batchids.push_back(unique_words[doc.words[ii]]);
    }
    
    MatrixXd Elogbeta_doc(m_Elogbeta.rows(),doc.words.size());
    for(int cc = 0; cc< doc.words.size(); cc++){
        Elogbeta_doc.col(cc) = m_Elogbeta.col(doc.words[cc]);
    }

    v=MatrixXd::Ones(2, K-1);
    v.row(1)=MatrixXd::Constant(1,K-1,alpha);
    VectorXd Elogsticks_2nd = expect_log_sticks(v); //K sized vector

    //Phi_j which is #word*#topic matrix refer to(eq17)
    MatrixXd phi = MatrixXd::Constant( doc.words.size(), K, 1.0/K);

    double likelihood = 0.0;
    double old_likelihood = -1e100;
    double converge = 1.0;
    double eps = 1e-100;

    MatrixXd mat_counts(1, doc.counts.size());
    for ( int cc = 0; cc < doc.counts.size(); cc++){
        mat_counts(0,cc) = doc.counts[cc];
    }
    int iter = 0;
    while( iter < 100 && ( converge < 0.0 || converge > var_converge )){
         //cout<<"\niter:"<<iter<<endl;
        //with array() is doc prod. without is matrix prod

        //##############var_phi######################
        // sum over words K*#wordis is scaled by word count
        MatrixXd Elogbeta_d_counts= Elogbeta_doc.array() 
                                    * mat_counts.replicate(Elogbeta_doc.rows(),1).array();
        //var phi is a K*T matrix. K-#document level topics T-# corpus level topics
        var_phi = phi.transpose() * Elogbeta_d_counts.transpose();


        if(iter > 2){
            //Elogsticks_1st T sized vector (T*1)
            //var_phi K*T
            var_phi = var_phi + Elogsticks_1st.transpose().replicate(var_phi.rows(),1);

        }
        //var_phi is K*T log norm is K sized vector
        MatrixXd log_norm = log_normalize(var_phi);
        MatrixXd log_var_phi = var_phi - log_norm.replicate(1, var_phi.cols());
        var_phi = exp(log_var_phi.array());

        //##############phi######################
        //phi is N*K N is the doc.words.size()
        MatrixXd tmpphi= var_phi * Elogbeta_doc; //var_phi is K*T sized Elogbeta_doc is T*N
        phi=tmpphi.transpose(); 

        if (iter > 2){
            //phi N*K Elogsticks_2nd is K size vector (K*1)
            phi = phi + Elogsticks_2nd.transpose().replicate(phi.rows(),1); 
        }
        log_norm = log_normalize(phi);
        //log_normi N*1  phi N*K
        MatrixXd log_phi = phi - log_norm.replicate(1,phi.cols());
        phi=exp(log_phi.array());


        //##############v######################
        //phi N*Ki
        //phi_all N*K mat_counts 1*N
        MatrixXd phi_all = phi.array() * mat_counts.transpose().replicate(1, phi.cols()).array();
        v.row(0)= 1.0 + phi_all.block(0,0,phi_all.rows(),K-1).colwise().sum().array();
        MatrixXd phi_all_colsum = phi_all.block(0,1,phi_all.rows(),K-1).colwise().sum();
        MatrixXd phi_all_inv_cumsum(1,K-1);
        phi_all_inv_cumsum(0,K-2) = phi_all_colsum(0,K-2);
        for (int k = 1; k< K-1; k++){
            phi_all_inv_cumsum(0,K-2-k) = phi_all_inv_cumsum(0,K-1-k) + phi_all_colsum(0,K-2-k);
        }

        v.row(1) = m_alpha + phi_all_inv_cumsum.array();
        Elogsticks_2nd = expect_log_sticks(v);


        //##################compute likelihood############
        likelihood = 0.0;
        //var_phi / c part
        //Elogsticks_1st T*1
        //log_var_phi K*T
        likelihood += ((Elogsticks_1st.transpose().replicate(var_phi.rows(),1) - log_var_phi).array() * var_phi.array()).sum();

        //v part //in the python code: m_K, m_alpha
        likelihood += (K-1)*alpha;
        //v is 2*(K-1)
        MatrixXd dig_sum = v.colwise().sum().unaryExpr(std::ptr_fun(digamma)); // 1*(K-1) sized
        MatrixXd t_t1 = MatrixXd::Ones(2, v.cols());
        t_t1.row(1) *= alpha;
        t_t1 -=v;
        MatrixXd t_t2 = v.unaryExpr(std::ptr_fun(digamma));
        t_t2 -= dig_sum.replicate(2,1);

        likelihood += (t_t1.array() * t_t2.array()).sum();

        MatrixXd tt1= v.colwise().sum().unaryExpr(std::ptr_fun(loggamma));
        MatrixXd tt2 = v.unaryExpr(std::ptr_fun(loggamma));

        likelihood -= tt1.sum() - tt2.sum();

        //z part
        //Elogsticks_2nd K*1
        likelihood += ((Elogsticks_2nd.transpose().replicate(phi.rows(),1) - log_phi).array() * phi.array()).sum();

        // x part, the data part
        //phi: N*K
        //Elog_beta_doc: T*N
        //var_phi: K*T
        //mat_counts: 1* N
        MatrixXd tt = Elogbeta_doc.array() * mat_counts.replicate(Elogbeta_doc.rows(),1).array();
        likelihood += (phi.transpose().array() * (var_phi * tt).array()).sum();
        
        converge = ( likelihood -old_likelihood) / abs(old_likelihood);
        old_likelihood = likelihood;

        if (converge < -0.000001){
            std::cout<<"warning, likelihood is decreasing!"<<std::endl;
        }
        iter ++;

            
    }
    //m_var_sticks_ss T*1
    ss.m_var_sticks_ss = ss.m_var_sticks_ss + (var_phi.colwise().sum()).transpose();
    
    MatrixXd t_s1 = phi.transpose().array() * mat_counts.replicate(phi.cols(),1).array();
    MatrixXd t_s2 = var_phi.transpose() * t_s1;
    for(int ii=0; ii<batchids.size(); ++ii){
        
        ss.m_var_beta_ss.col(batchids[ii]) = ss.m_var_beta_ss.col(batchids[ii]) + t_s2.col(ii);
    }
    
    return likelihood;
}

 void model::update_lambda( suff_stats & sstats, std::vector<int> & word_list, int opt_o){

     m_status_up_to_date = false;
     if(word_list.size()==W){ //seen all the vocabulary
         m_status_up_to_date = true;
     }
     // rhot will be between 0 and 1, and says how much to weight
     //the information we got from this mini-batch.
     
     double rhot = scale * pow( m_tau+ m_updatect, -kappa);
     if(rhot < rhot_bound){
         m_rhot = rhot_bound;
     }else{
         m_rhot = rhot;
     }

     //Update appropriate columns of lambda based on documents.

     assert(word_list.size() == sstats.m_var_beta_ss.cols());
     assert(m_lambda.rows() == sstats.m_var_beta_ss.rows());
     for (int ww=0; ww < word_list.size(); ww++){
         //cheng: lambada = (1-rhot)*lambda + rhot * ~lambda
         // eq 25 in Chong s paper
         m_lambda.col(word_list[ww]) = m_lambda.col(word_list[ww]) * (1-rhot) +
         rhot * sstats.m_var_beta_ss.col(ww) * D / sstats.m_batchsize;
     }
     m_lambda_sum = m_lambda. rowwise().sum();

     m_updatect +=1;
     for (int ww=0; ww < word_list.size(); ww++){
        m_timestamp(word_list[ww]) = m_updatect;
     }
     m_r.push_back(m_r.back() + log(1-rhot));
    //m_varphi_ss is a T sized vector. 
     assert(m_varphi_ss.rows() == sstats.m_var_sticks_ss.rows());
     m_varphi_ss  = ( 1.0 -rhot) * m_varphi_ss + rhot* sstats.m_var_sticks_ss * D / sstats.m_batchsize;

    if(opt_o){
        optimal_ordering();
    }

    //uodate top level sticks
    MatrixXd var_sticks_ss = MatrixXd::Zero(2, T-1);
    //Eq(26) Chong
    m_var_sticks.row(0) = m_varphi_ss.head(T-1) + VectorXd::Ones(T-1);
    //Eq(27) Chong
    MatrixXd inv_sum_varphi(1,T-1);
    inv_sum_varphi(0,T-2) = m_varphi_ss(T-2);
    for( int tt=1; tt<T-1; tt++){
        inv_sum_varphi(0, T-2-tt) = inv_sum_varphi(0, T-1-tt) + m_varphi_ss(T-2-tt);
    }
    m_var_sticks.row(1) = inv_sum_varphi.array() + gamma;

     



 }

 void model::process_documents(std::vector<int> & doc_unseen, double & score, int & count, double & unseen_score, int & unseen_count){
    
     //cout<<"\t\tIn process_documents..."<<endl;
     m_num_docs_parsed += batch_docs.size();
     adding_noise = 0;
     adding_noise_point = min_adding_noise_point;

     if(m_adding_noise){
     }

     std::unordered_map<int, int> unique_words;
     std::vector<int> word_list;

     if(adding_noise){
     }else{
         for(int dd=0; dd< batch_docs.size(); dd++){
             document ddoc=batch_docs[dd];
             for(int ww =0 ; ww < ddoc.length; ww++){
                 int w = ddoc.words[ww];
                 auto it = unique_words.begin();
                 it = unique_words.find(w);
                 if( it == unique_words.end() ){
                     // it is not in the unique_words
                     int uws=unique_words.size();
                     unique_words[w] = uws;
                     word_list.push_back(w);
                 }
                }
         }
     }
     int Wt = word_list.size(); //V
     for(int tt=0; tt < word_list.size(); tt++){
     //  the lazy updates on the necessart columns of lambda
//        cout<<"m_r size:"<<m_r.size()<<endl;
  //      cout<<"time stamp size"<<m_timestamp.cols()<<endl;
   //     cout<<"time_stamp_wl_tt:"<<m_timestamp(word_list[tt])<<endl;
         int rw = m_r[m_timestamp(word_list[tt])];
     //    cout<<"\t\t\t rw:"<< rw<<endl;
         m_lambda.col(word_list[tt]) = m_lambda.col(word_list[tt]).array()* exp(m_r.back() - rw);
       //  cout<<"done update lambda"<<endl;
         // update_Elogbeta
         //beta here is the beta in the online LDA eq(6) which is the topic-words distribution 
         //E[log beta_{kw} ] = psi(lambda) -psi(sum lambda)
       //  cout<<"1:"<<m_Elogbeta.col(word_list[tt]).rows()<<endl;
       //  cout<<"2:"<<m_lambda.col(word_list[tt]).rows()<<endl;
       //  cout<<"3:"<<m_lambda_sum.rows()<<endl;
         m_Elogbeta.col(word_list[tt]) = (m_eta + m_lambda.col(word_list[tt]).array()).unaryExpr(std::ptr_fun(digamma)) - (W*m_eta + m_lambda_sum.array()).unaryExpr(std::ptr_fun(digamma));
       //  cout<<"done update Elogbeta"<<endl;
     }


     suff_stats ss(m_T, Wt, batch_docs.size());

     Elogsticks_1st = expect_log_sticks( m_var_sticks);

     // run variational inference on some new docs
     score =0.0;
     count = 0;
     unseen_score = 0.0;
     unseen_count = 0;
     double doc_score;
     for( int dd=0; dd < batch_docs.size(); dd++ ){
         //cout<<"dd:"<<dd<<endl;
         document doc = batch_docs[dd];
         doc_score = doc_e_step(doc, ss, Elogsticks_1st, word_list, unique_words);
         count += doc.total;
         score += doc_score;
            
         std::vector<int>::iterator p;
         p = find ( doc_unseen.begin(), doc_unseen.end() , dd);
         if (p != doc_unseen.end()){
             unseen_score += doc_score;
             unseen_count += doc.total;
         }
     }

     int opt_o = 1;
     update_lambda(ss, word_list, opt_o);
 }

 void model::optimal_ordering(){
     //m_lambda_sum is a T sized vector
        std::vector<int> idx;  
        //m_lambda_sum is a T sized vector
        for( int ii=0; ii<m_lambda_sum.rows(); ii++){
            idx.push_back(ii);
        }
        sort(idx.begin(), idx.end(), index_cmp(m_lambda_sum));
        VectorXd m_varphi_ss_tmp = m_varphi_ss; // T sized
        MatrixXd m_lambda_tmp = m_lambda; //T*W sized
        VectorXd m_lambda_sum_tmp = m_lambda_sum;//T*W sized
        MatrixXd m_Elogbeta_tmp = m_Elogbeta; //T*W sized

        assert(idx.size()== m_Elogbeta_tmp.rows());
        assert(idx.size()== m_varphi_ss_tmp.rows());

        for(int ii=0; ii< idx.size(); ii++){
            m_varphi_ss(ii) = m_varphi_ss_tmp(idx[ii]);
            m_lambda.row(ii) = m_lambda_tmp.row(idx[ii]);
            m_lambda_sum(ii) = m_lambda_sum_tmp(idx[ii]);
            m_Elogbeta.row(ii) = m_Elogbeta_tmp.row(ii); 

        }
 }

 void model::update_expectations(){

     /*Since we're doing lazy updates on lambda, at any given moment 
        the current state of lambda may not be accurate. This function
        updates all of the elements of lambda and Elogbeta so that if (for
        example) we want to print out the topics we've learned we'll get the
        correct behavior.*/

        for ( int ww= 0; ww < W; ww++){
            //m_r<<m_r(m_r.tail(1)) + log(1-rhot);
            //m_timestamp(word_list[ww]) = m_updatect;
            m_lambda.col(ww) *= exp(m_r.back()- m_r[m_timestamp(ww)]);
        }
        //m_lambda: T*W matrix m_lambda_sum: T sized vector
        MatrixXd tt1=(m_eta + m_lambda.array()).unaryExpr(std::ptr_fun(digamma)); 
        MatrixXd tt2=( W*m_eta + m_lambda_sum.array()).unaryExpr(std::ptr_fun(digamma));

        m_Elogbeta = tt1 - tt2.replicate(1, m_lambda.cols());
        assert(m_timestamp.rows() == W);
        m_timestamp= MatrixXd::Constant(W,1,m_updatect);
        m_status_up_to_date = true;

 }

void model::save_options( std::string option_file){

    std::ofstream output;
    output.open(option_file.c_str());

    output<<"T:\t"<<T<<"\n";
    output<<"K:\t"<<K<<"\n";
    output<<"D:\t"<<D<<"\n";
    output<<"W:\t"<<W<<"\n";
    output<<"eta:\t"<<eta<<"\n";
    output<<"alpha:\t"<<alpha<<"\n";
    output<<"gamma:\t"<<gamma<<"\n";
    output<<"kappa:\t"<<kappa<<"\n";
    output<<"tau:\t"<<tau<<"\n";
    output<<"batchsize:\t"<<batchsize<<"\n";
    output<<"max_time:\t"<<max_time<<"\n";
    output<<"max_iter:\t"<<max_iter<<"\n";
    output<<"var_converge:\t"<<var_converge<<"\n";
    output<<"corpus name:\t"<<corpus_name<<"\n";
    output<<"date_path:\t"<<data_path<<"\n";
    output<<"test date_path:\t"<<test_data_path<<"\n";
    output<<"directory:\t"<<directory<<"\n";
    output<<"save_lag:\t"<<save_lag<<"\n";
    output<<"pass ratio:\t"<<pass_ratio<<"\n";
    output<<"new_init:\t"<<new_init<<"\n";
    output<<"scale:\t"<<scale<<"\n";
    output<<"adding_noise:\t"<<adding_noise<<"\n";
    output<<"seq_mode:\t"<<seq_mode<<"\n";
    output<<"fixed_lag:\t"<<fixed_lag<<"\n";

    output.close();
}

void model::save_model( std::string res_directory, int total_doc_count){
    //save topics
    std::cout<<"save topics..."<<std::endl;
    if( m_status_up_to_date == false ){
        update_expectations();
    }
    std::string topicfilename = res_directory + "/doc_count_" 
    + convert(total_doc_count)  +".topics";

    std::ofstream topic_file;
    topic_file.open(topicfilename.c_str());
    MatrixXd betas = m_lambda.array() + m_eta; //T*W matirx

    for( int tt = 0; tt < betas.rows(); tt++){
        for ( int nn=0; nn < betas.cols(); nn++){

            topic_file<<betas(tt,nn)<<" ";

        }
        topic_file<<"\n";
    }

    topic_file.close();


    // save other part
    //save first level stick
    std::cout<<"save Elogstick_1st..."<<std::endl;
    std::string elogstick1_filename = res_directory + "/doc_count_" + 
    boost::lexical_cast<std::string>(total_doc_count) +".elogstick1st";

    std::ofstream elogstick1_file;
    elogstick1_file.open(elogstick1_filename.c_str());
    for ( int tt = 0; tt< Elogsticks_1st.rows(); tt++){
        elogstick1_file<<Elogsticks_1st(tt)<<" ";
    }
    elogstick1_file.close();

    // save lda_alpha
    std::cout<<"save lda_alpha..."<<std::endl;
    std::string alphafilename = res_directory + "/doc_count_" + 
    boost::lexical_cast<std::string>(total_doc_count) +".lda_alpha";

    std::ofstream alpha_file;
    alpha_file.open(alphafilename.c_str());
    for ( int tt = 0; tt< lda_alpha.rows(); tt++){
        alpha_file<<lda_alpha(tt)<<" ";
    }
    alpha_file.close();
}   

void model::hdp_to_lda(){

    if(m_status_up_to_date == false){
        update_expectations();
    }

    //m_var_sticks are 2*T-1
    MatrixXd sticks = m_var_sticks.row(0).array() / (m_var_sticks.row(0) + m_var_sticks.row(1)).array();

    lda_alpha = VectorXd::Zero(T);
    double left = 1.0;
    for( int i  =0; i<T-1; ++i){
        lda_alpha(i)= sticks(i)*left;
        left = left - lda_alpha(i);
    }
    lda_alpha(T-1) = left;
    lda_alpha = lda_alpha.array() * alpha;

    // m_lambda_sum: T sized Vectot
    VectorXd tt2 = W*m_eta +  m_lambda_sum.array();
    lda_beta = (m_lambda.array() + m_eta) / tt2.replicate(1,m_lambda.cols()).array();

}

void model::lda_e_step(document & doc, double & likelihood){

    lda_gamma = VectorXd::Ones(lda_alpha.rows()); // T sized vector
    MatrixXd Elogtheta = dirichlet_expectation(lda_gamma); // T sized 
    MatrixXd expElogtheta = exp(Elogtheta.array()); //Tsized
    MatrixXd betad(T,doc.words.size() ); // T*N sized

    VectorXd mat_counts(doc.counts.size()); //N*1
    for (int n=0; n< doc.words.size(); ++n){
        betad.col(n) = lda_beta.col(doc.words[n]);
        mat_counts(n) = doc.counts[n];
    }

    MatrixXd phinorm = (expElogtheta.transpose()* betad).array() + 1e-100;// (1*T) *(T*N) = 1*N

    int iter = 0;
    VectorXd last_gamma;
    double meanchangethresh = 0.01;
    // set the max iter to 100 as chong
    while(iter < 100){
        last_gamma = lda_gamma;
        iter++;
        likelihood = 0.0;
        MatrixXd tt =  mat_counts.array()/phinorm.transpose().array(); // N*1 
        //betad T*N
        assert(lda_alpha.rows() == expElogtheta.rows());
        lda_gamma = lda_alpha.array() + expElogtheta.array()* ( tt.transpose() * betad.transpose()).transpose().array();
        Elogtheta = dirichlet_expectation(lda_gamma);
        expElogtheta = exp(Elogtheta.array());

        phinorm = (expElogtheta.transpose()* betad).array() + 1e-100;// (1*T) *(T*N) = 1*N

        VectorXd change=abs((lda_gamma.array()-last_gamma.array()).array());
        double meanchange = change.sum() / change.rows();
        if ( meanchange < meanchangethresh ){
            break;
        }
    }
    likelihood = (mat_counts.array() * log(phinorm.transpose().array()).array()).sum();

    // E[log p(theta | alpha ) - log q(theta | gamma )
    likelihood += ((lda_alpha - lda_gamma).array() * Elogtheta.array()).sum();
    likelihood += (lda_gamma.unaryExpr(std::ptr_fun(loggamma)) - lda_alpha.unaryExpr(std::ptr_fun(loggamma))).sum();
    likelihood += loggamma(lda_alpha.sum()) - loggamma(lda_gamma.sum());

    
}

void model::lda_e_step_split(document & doc, double & score){

    //split the document
    int num_train=ceil(doc.length / 2.0); // even numbers
    int num_test = floor( doc.length /2.0);

    VectorXd words_train(num_train);
    VectorXd counts_train(num_train);
    VectorXd words_test(num_test);
    VectorXd counts_test(num_test);
    int ii=0;
    int jj=0;
    for( int i=0; i< doc.length; ++i){
        if(i%2 == 0){
            words_train(ii)=doc.words[i];
            counts_train(ii)= doc.counts[i];
            ii++;
        }else{
            words_test(jj) = doc.words[i];
            counts_test(jj) = doc.counts[i];
            jj++;
        }
    }
    
    //do lda e step on the train part
    //the same as online lda alpgorithm
    lda_gamma = VectorXd::Ones(lda_alpha.rows()); // T sized vector
    MatrixXd Elogtheta = dirichlet_expectation(lda_gamma); // T sized 
    MatrixXd expElogtheta = exp(Elogtheta.array()); //Tsized
    MatrixXd betad(T, words_train.rows() ); // T*N sized

    VectorXd mat_counts(counts_train.rows()); //N*1
    for (int n=0; n< words_train.rows(); ++n){
        betad.col(n) = lda_beta.col(words_train(n));
        mat_counts(n) = doc.counts[n];
    }

    // the optimal phi_{dwk} is proportional to 
    //expElogtheta_k * expElogbetad_w. Phinorm is the normalizer
    MatrixXd phinorm = (expElogtheta.transpose()* betad).array() + 1e-100;// (1*T) *(T*N) = 1*N

    int iter = 0;
    VectorXd last_gamma;
    double meanchangethresh = 0.01;;
    // set the max iter to 100 as chong
    while(iter < 100){
        last_gamma = lda_gamma;
        iter++;
        double likelihood = 0.0;
        MatrixXd tt =  mat_counts.array()/phinorm.transpose().array(); // N*1
        // this way to present phi implicitly is to save memory and time
        //substituting the value of the optimal ohi back into
        // the uopdate for gamma gives this update. Cf. Lee&Seung 2001
        lda_gamma = lda_alpha.array() + expElogtheta.array()*
            ( tt.transpose() * betad.transpose()).transpose().array();
        Elogtheta = dirichlet_expectation(lda_gamma);
        expElogtheta = exp(Elogtheta.array());
        phinorm = (expElogtheta.transpose()* betad).array() + 1e-100;// (1*T) *(T*N) = 1*N

        VectorXd change=abs((lda_gamma.array()-last_gamma.array()).array());
        double meanchange = change.sum() / change.rows();

        if ( meanchange < meanchangethresh ){
            break;
        }
    }
    lda_gamma = lda_gamma.array()/ lda_gamma.sum();
    mat_counts = counts_test;
    count_split = mat_counts.sum();

    MatrixXd betad_tst(T, words_test.rows() ); // T*N sized
    for (int n=0; n< words_test.rows(); ++n){
        betad_tst.col(n) = lda_beta.col(words_test(n));
    }
    score = (mat_counts.transpose().array() * log(((lda_gamma.transpose()* betad_tst).array() + 1e-100).array())).sum();
}

}
