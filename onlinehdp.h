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
 * along with onlineHDP_C++; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA.
 */

#ifndef ONLINEHDP_H
#define ONLINEHDP_H

#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>
#include <iterator>
#include <Eigen/Dense>
#include "corpus.h"
#include <unordered_map>

namespace HDP {

//sufficient statistic 
class suff_stats {
public:
    int m_batchsize;
    Eigen::VectorXd m_var_sticks_ss; //T sized vector
    Eigen::MatrixXd m_var_beta_ss; // T *N

    suff_stats(int T, int Wt, int Dt)
        :   m_batchsize(Dt) {
        m_var_sticks_ss = Eigen::VectorXd::Zero(T);
        m_var_beta_ss = Eigen::MatrixXd::Zero(T, Wt);
    }

    void set_zero() {
        m_var_sticks_ss = Eigen::VectorXd::Zero(m_var_sticks_ss.size());
        m_var_beta_ss = Eigen::MatrixXd::Zero(m_var_beta_ss.rows() , m_var_beta_ss.cols() );
    }
};

// model and onlinehdp
class model {
public:
    int T; // top level truncation
    int K; // second level truncation
    int D; //number of documents
    int W; //as W in python; size of vocabulary
    double eta; // the topic/words Dirichlet parameter
    double alpha; // the document level beta parameter 
    double gamma; // the corpus level beta parameter
    double kappa; //learning rate which is a parameter to compute rho
    double tau; // slow down which is a parameter to compute rho
    int batchsize;
    int max_time; // max time to run training in seconds
    int max_iter; // max iteration to run training
    double var_converge; // relative chane on doc lower bound
    int random_seed;
    std::string corpus_name;
    std::string data_path;
    std::string test_data_path;
    std::string directory; // result directory
    int save_lag;
    double pass_ratio;
    int new_init;
    double scale;//used to compute rho. default 1
    int adding_noise;
    int seq_mode;
    int fixed_lag;


    // the sticks
    Eigen::MatrixXd m_var_sticks; // 2*T-1
    Eigen::VectorXd m_varphi_ss; // T sized vector
    Eigen::MatrixXd m_lambda; // T * W matrix
    Eigen::VectorXd m_lambda_sum; // sum over W, result a T sized vecor
    Eigen::MatrixXd m_Elogbeta; //T*W
    Eigen::VectorXd Elogsticks_1st; // T sized vector

    Eigen::MatrixXd var_phi; //K*T

    double m_tau;
    double m_eta;
    double m_alpha; // alpha is updated //check the code
    double m_adding_noise;
    int m_T;
    int min_adding_noise_point;
    int adding_noise_point;
    double min_adding_noise_ratio;
    int m_updatect; // used in compue rho
    bool m_status_up_to_date;
    int m_num_docs_parsed;

    Eigen::MatrixXd v; //the beta parameter eq(5),(6) 
                // 2* (K-1) sized
   
    std::vector<int> doc_seen;
    std::vector<int> doc_unseen;

    double m_rhot;
    double rhot_bound;


    Eigen::VectorXd lda_alpha; //T sized Vector
    Eigen::MatrixXd lda_beta; //T *W sized Matrixi

    Eigen::VectorXd lda_gamma;
    int count_split; //return from lda_e_step_split
    
    
    
    // docs in a batch
    std::vector<document> batch_docs;
    //timestamps and normalizers for lazt updates
    Eigen::VectorXd m_timestamp; //W sized vetcor
    std::vector<int> m_r; 
public:
    //results
    void init_hdp();
    void run_online_hdp();
    // For a vector theta ~ Dir(alpha), computes E[log(theta)] given alpha.
    Eigen::MatrixXd dirichlet_expectation(const Eigen::MatrixXd & alpha);

    //For stick-breaking hdp, this returns the E[log(sticks)] 
    Eigen::VectorXd expect_log_sticks(const Eigen::MatrixXd & sticks);
   // e step for a single document 
    double doc_e_step( document & doc,suff_stats & ss,const  Eigen::VectorXd & Elogsticks, std::vector<int> & word_list, std::unordered_map<int, int> & unique_words);
    
    void update_lambda(suff_stats & sstats, std::vector<int> &word_list, int opt_o);
    
    void process_documents(std::vector<int> & doc_unseen, double &  score, int & count,  double & unseen_score, int & unseen_count);

    void optimal_ordering();

    void update_expectations();

    void save_options(std::string option_file);
    void save_model(std::string res_directory, int total_doc_count);
    
    void hdp_to_lda();

    void lda_e_step(document & doc, double & likelihood);

    void lda_e_step_split(document & doc, double & score);

    friend int parse_args(int argc,char** argv,model &model);
};

struct index_cmp
{
index_cmp(const Eigen::VectorXd &pArr)
:arr(pArr){}
bool operator()(const size_t a,const size_t b) const
{
return arr[a] > arr[b];
}
const Eigen::VectorXd arr;
};

}

#endif
