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
#include "utils.h"
#include <math.h>

using namespace Eigen;

namespace HDP {

int parse_args(int argc, char ** argv , model & pmodel){

    //init values

    int T=-1; // top level truncation
    int K=-1; // second level truncation
    int D=-1; //number of documents
    int W=-1; //as W in python; size of vocabulary
    double eta=-1; // the topic/words Dirichlet parameter
    double alpha=-1; // the document level beta parameter 
    double gamma=-1; // the corpus level beta parameter
    double kappa=-1; //learning rate which is a parameter to compute rho
    double tau=-1; // slow down which is a parameter to compute rho
    int batchsize=-1;
    int max_time=-1; // max time to run training in seconds
    int max_iter=-1; // max iteration to run training
    double var_converge=0.0001; // relative chane on doc lower bound
    int random_seed=999931111;
    std::string corpus_name;
    std::string data_path;
    std::string test_data_path;
    std::string directory; // result directory
    int save_lag=500;
    double pass_ratio = 0.5;
    int new_init=0;
    double scale=1.0;
    int adding_noise=0;
    int seq_mode=0;
    int fixed_lag=0;

    int  i=1;
    while (i < argc){
        std::string arg = argv[i];
        if (arg == "-t") {
            T = atoi(argv[++i]);

        }else if ( arg == "-k") {
            K = atoi(argv[++i]);

        }else if ( arg == "-d") {
            D = atoi(argv[++i]);

        }else if ( arg == "-w") {
            W = atoi(argv[++i]);

        }else if ( arg == "-eta") {
            eta = atof(argv[++i]);

        }else if ( arg == "-alpha") {
            alpha = atof(argv[++i]);

        }else if ( arg == "-gamma") {
            gamma = atof(argv[++i]);

        }else if ( arg == "-kappa") {
            kappa = atof(argv[++i]);

        }else if ( arg == "-tau") {
            tau = atof(argv[++i]);

        }else if ( arg == "-batchsize") {
            batchsize = atoi(argv[++i]);

        }else if ( arg == "-max_time") {
            max_time = atoi(argv[++i]);

        }else if ( arg == "-max_iter") {
            max_iter = atoi(argv[++i]);

        }else if ( arg == "-var_converge") {
            var_converge = atof(argv[++i]);

        }else if ( arg == "-random_seed") {
            random_seed = atoi(argv[++i]);

        }else if ( arg == "-corpus_name") {
            corpus_name = argv[++i];

        }else if ( arg == "-data_path") {
            data_path = argv[++i];

        }else if ( arg == "-test_path") {
            test_data_path = argv[++i];

        }else if ( arg == "-res") {
            directory = argv[++i];

        }else if ( arg == "-lag") {
            save_lag = atoi(argv[++i]);

        }else if ( arg == "-ratio") {
            pass_ratio = atof(argv[++i]);

        }else if ( arg == "-new_init") {
            new_init = atoi(argv[++i]);

        }else if ( arg == "-scale") {
            scale = atoi(argv[++i]);

        }else if ( arg == "-adding_noise") {
            adding_noise = atoi(argv[++i]);

        }else if ( arg == "-seq_mode") {
            seq_mode = atoi(argv[++i]);

        }else if ( arg == "-fixed_lag") {
            fixed_lag = atoi(argv[++i]);

        }else {
            std::cout << "i don't know parameter " << arg << "\n";
            return 1;
        }

        i++;
    }

    std::cout<<"read in arguments done"<<std::endl;

    std::cout<<"start to assign values to the model"<<std::endl;
    
    if ( T == -1 ){
        std::cout<<"pls specify top level truncation"<<std::endl;
        return 1;
    }else{
        pmodel.T = T;
    }
 
    if ( K == -1 ){
        std::cout<<"pls specify second level truncation"<<std::endl;
        return 1;
    }else{
        pmodel.K = K;
    }
    
    if ( D == -1 ){
        std::cout<<"pls specify the number of documents"<<std::endl;
        return 1;
    }else{
        pmodel.D = D;
    }
    
    if ( W == -1 ){
        std::cout<<"pls specify the vocaubulary size"<<std::endl;
        return 1;
    }else{
        pmodel.W = W;
    }
    
    if ( eta == -1 ){
        std::cout<<"pls specify the topic_words Dirichlet parameter"<<std::endl;
        return 1;
    }else{
        pmodel.eta = eta;
    }
    
    if ( alpha == -1 ){
        std::cout<<"pls specify document level beta parameter alpha"<<std::endl;
        return 1;
    }else{
        pmodel.alpha = alpha;
    }

    if ( gamma == -1 ){
        std::cout<<"pls specify corpus level beta prarameter gamma"<<std::endl;
        return 1;
    }else{
        pmodel.gamma = gamma;
    }

    if ( kappa == -1 ){
        std::cout<<"pls specify learning rate kappa"<<std::endl;
        return 1;
    }else{
        pmodel.kappa = kappa;
    }

    if ( tau == -1 ){
        std::cout<<"pls specify slow prarameter tau"<<std::endl;
        return 1;
    }else{
        pmodel.tau = tau;
    }

    if ( batchsize == -1 ){
        std::cout<<"pls specify batch size"<<std::endl;
        return 1;
    }else{
        pmodel.batchsize = batchsize;
    }

    if ( max_time == -1 ){
        std::cout<<"pls specify max time"<<std::endl;
        return 1;
    }else{
        pmodel.max_time = max_time;
    }

    if ( max_iter == -1 ){
        std::cout<<"pls specify max iter"<<std::endl;
        return 1;
    }else{
        pmodel.max_iter = max_iter;
    }

    pmodel.var_converge = var_converge;

    pmodel.random_seed = random_seed;

    if ( corpus_name == "" ){
        std::cout<<"pls specify corpus name"<<std::endl;
        return 1;
    }else{
        pmodel.corpus_name = corpus_name;
    }

    if ( data_path == ""){
        std::cout<<"pls specify data path"<<std::endl;
        return 1;
    }else{
        pmodel.data_path = data_path;
    }
    if ( test_data_path == ""){
        std::cout<<"pls specify test data path"<<std::endl;
        return 1;
    }else{
        pmodel.test_data_path = test_data_path;
    }

    if ( directory == ""){
        std::cout<<"pls specify result directory"<<std::endl;
        return 1;
    }else{
        pmodel.directory = directory;
    }

    pmodel.save_lag = save_lag;
    pmodel.pass_ratio = pass_ratio;
    pmodel.new_init = new_init;
    pmodel.scale = scale;
    pmodel.adding_noise = adding_noise;
    pmodel.seq_mode = seq_mode;
    pmodel.fixed_lag = fixed_lag;

    std::cout<<"parsing arguments done"<<std::endl;
    return 0;
}


int getdir(std::string dir, std::vector<std::string> &files ){
    DIR *dp;
    struct dirent *dirp;
    if((dp  = opendir(dir.c_str())) == NULL) {
        std::cout << "Error(" << errno << ") opening " << dir << std::endl;
        return errno;
    }

    while ((dirp = readdir(dp)) != NULL) {
        files.push_back(std::string(dirp->d_name));
    }
    closedir(dp);
    return 0;
}

MatrixXd log_normalize( const  MatrixXd & v){
    // compute log(sum(exp(v))) return log_norm
    //return v and log_norm in python code

    double log_max = 100.0;
    VectorXd log_norm;
    if(v.cols()==1){ // VectorXd in this case
        double log_shift = log_max - log(v.rows()+1.0) - v.maxCoeff();
        VectorXd sum_tmp = v.array() + log_shift;
        VectorXd exp_tmp=exp(sum_tmp.array());
        double tot= exp_tmp.sum();
        log_norm(0)=log(tot)-log_shift;


    }else{
        double vlog=log(v.cols()+1);
        MatrixXd log_shift(v.rows(),1);

        for(int rr=0; rr<v.rows(); rr++){
            double max_val_r = v.row(rr).maxCoeff();
            log_shift(rr,0) = log_max -vlog -max_val_r;
        }

        MatrixXd sum_tmp = v.array() + log_shift.replicate(1,v.cols()).array();
        MatrixXd exp_tmp = exp(sum_tmp.array());
        MatrixXd tot = exp_tmp.rowwise().sum();
        MatrixXd logtot=log(tot.array());
        log_norm = logtot - log_shift; // a VectorXd

    }

    return log_norm;
}
double digamma(double x){
    double y = boost::math::digamma(x);
    return y;
}
double loggamma(double x){
    double y = boost::math::lgamma(x);
    return y;
}

}
