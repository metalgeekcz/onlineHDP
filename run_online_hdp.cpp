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
#include "onlinehdp.h"
#include "corpus.h"
#include <algorithm>
#include <random>
#include <ctime>
#include <cstdlib>
#include <vector>
#include <sys/types.h>
#include <sys/stat.h>
#include <time.h>
#include <dirent.h>

using namespace Eigen;
using namespace HDP;

// class generator:
struct c_unique {
    int current;
    c_unique() {current=0;}
    int operator()() {return ++current;}
} UniqueNumber;

int main(int argc, char ** argv) {

    model hdp;

    // parse args
    std::cout<<"start parse args"<<std::endl;
    if (parse_args(argc, argv, hdp)){
        std::cout<<"parse_arg error"<<std::endl;
        return 1;
    }

    //training data
    corpus c_train; 
    
    std::vector<std::string> train_filenames; // only used in non-seq mode, store all file splits
    int num_train_splits;
    int cur_chosen_split;
    std::string cur_train_filename;
    std::ifstream input_file;
    if( hdp.seq_mode){
        std::cout<<"seq mode open file..."<<std::flush;
        input_file.open(hdp.data_path.c_str());
        std::cout<<"done"<<std::endl;
    }else{
         DIR * dirp = opendir(hdp.data_path.c_str());
         while(dirent *dp=readdir(dirp)){
             struct stat filestat;
             if(stat( dp->d_name, &filestat ))
                 train_filenames.push_back(dp->d_name);
         }

         num_train_splits = train_filenames.size();
         cur_chosen_split = 0;
         cur_train_filename = hdp.data_path + train_filenames[cur_chosen_split];
         //c_train read data for the current split
         c_train.read_data(cur_train_filename);
    }

    corpus c_test;
    int c_test_word_count=0;

    if(hdp.test_data_path != ""){
        //std::cout<<"reading test data ..."<<flush;
        c_test.read_data(hdp.test_data_path);
        //std::cout<<"done"<<std::endl;
        for (int dd=0; dd< c_test.docs.size(); dd++){
            c_test_word_count += c_test.docs[dd].total;
        }
    }

    //result directory
    std::string res_directory = hdp.directory + "/" + hdp.corpus_name+"_kappa_" + convert(hdp.kappa)
        + "_tau_" + convert(hdp.tau) + "_batchsize_" + convert(hdp.batchsize);
    //creat a directory with read/write/search permissions for owner and group, and with read/search permissions for others.
    std::cout<<"creat result directory: "<<res_directory<<"..."<<std::flush;
    mkdir(res_directory.c_str(),  S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    std::cout<<"done"<<std::endl;

    //save option file
    std::string  options_file = res_directory + "/options.dat";
    std::cout<<"save options to:"<< options_file<<"..."<<std::flush;
    hdp.save_options(options_file);
    std::cout<<"done"<<std::endl;

    //initial the model 
    std::cout<<"initial the model..."<<std::flush;
    hdp.init_hdp();
    std::cout<<"done"<<std::endl;

    //setting up counters and log files
    int iter=0;
    int save_lag_counter=0;
    int total_doc_count = 0;
    int split_doc_count = 0;
    std::string log_file = res_directory + "/log.dat";
    std::ofstream log_file_s(log_file.c_str());
    log_file_s<<"iteration time doc.count score word.count unseen.score unseen.word.count\n";
    std::string test_log_file = res_directory + "test-log.dat";
    std::ofstream test_log_file_s;
    if( hdp.test_data_path != ""){
        test_log_file = res_directory + "/test_log.dat";
        test_log_file_s.open(test_log_file.c_str());
        test_log_file_s<<"iteration time doc.count score word.count unseen.score unseen.word.count\n";
    }

    //variational inference
    //we only need to shuffle it once
    std::vector<int> split_range(c_train.docs.size());
    if(hdp.seq_mode==0){
        //generate natural number seq range (1~ num_doc)
        generate (split_range.begin(), split_range.end(), UniqueNumber);
        //shuffle the numbers
        random_shuffle(split_range.begin(), split_range.end());
    }

    int c_test_word_count_split;
    double test_score;
    double test_score_split;
    double score;
    double likelihood;
    int count;
    double unseen_score;
    int unseen_count;

    clock_t t0;
    clock_t total_time=0;

    int split_iter=0;

    while(1){
        iter ++;
        split_iter ++;
        if( iter%50== 1 ){
            std::cout << "iteration: "<<iter <<std::endl;
        }

        t0 = clock();

        //sample the documents
        srand( unsigned (time(0)));   

        if(hdp.seq_mode){
           c_train.read_stream_data(input_file, hdp.batchsize);
           hdp.batchsize = c_train.num_docs; 
           //std::cout<<"num of docs in c_train: "<< c_train.num_docs;
           if(hdp.batchsize ==0){
               std::cout<<"batch size 0, break"<<std::endl;
               break;
           }
           hdp.batch_docs = c_train.docs;
           //doc_unseen is unseen_ids in python
           hdp.doc_unseen.clear();
           for (int dd=0; dd<hdp.batchsize; ++dd){
               hdp.doc_unseen.push_back(dd);
           }

           
        }else{
            // sample batchsized documents from current split
            //std::cout<<"sample batchsized documents from current split"<<std::endl;
            std::vector<int> ids(hdp.batchsize);
            copy(split_range.begin()+ hdp.batchsize*(split_iter-1), split_range.begin()+hdp.batchsize * split_iter, ids.begin());
            //batch_docs is the docs in python
            hdp.batch_docs.clear(); // clean docs from last batch
            for( int ii=0; ii<hdp.batchsize; ++ii){
                hdp.batch_docs.push_back(c_train.docs[ids[ii]]);
            }
            /*hdp.doc_seen.reserve(hdp.batchsize * split_iter); // it is a set in python
            hdp.doc_unseen.clear();
            hdp.doc_unseen.reserve(c_train.num_docs-hdp.batchsize * split_iter);

            copy(split_range.begin(), split_range.begin()+hdp.batchsize * split_iter, hdp.doc_seen.begin());
            copy(split_range.begin()+hdp.batchsize * split_iter+1, split_range.end(), hdp.doc_unseen.begin());*/
            
           //since we just shuffle it once and read next batchsize docs in each iter 
           hdp.doc_unseen.clear();
           for (int dd=0; dd<hdp.batchsize; ++dd){
               hdp.doc_unseen.push_back(dd);
           }
        }

        total_doc_count += hdp.batchsize;
        split_doc_count += hdp.batchsize;
        //std::cout<<"total_doc_count: "<< total_doc_count<<std::endl;
        //std::cout<<"split_doc_count: "<<split_doc_count<<std::endl;
        //Do online inference and evaluate on the fly dataset
        //std::cout<<"\t process documents..."<<std::endl;
        hdp.process_documents(hdp.doc_unseen, score, count, unseen_score, unseen_count);
        total_time += clock()-t0;
        std::string linetow=convert(iter)+"\t" + convert(total_time) + "\t" 
            + convert(total_doc_count) + "\t"
            + convert(score) + "\t" + convert(count) + "\t" 
            + convert(unseen_score)+"\t" + convert(unseen_count) + "\n";
            //std::cout<<linetow;
        log_file_s<<linetow;
        // Evaluate on the test data: fixed and folds
        //std::cout<<"total_doc_count:"<<total_doc_count<<" save_lag:"<<hdp.save_lag<<std::endl;
        if(total_doc_count % hdp.save_lag ==0){
            if( hdp.fixed_lag==0 && save_lag_counter <10){
                save_lag_counter += 1;
                    hdp.save_lag = hdp.save_lag *2;
            }

            hdp.save_model(res_directory, total_doc_count);//include python save topics and cPickle,dump

            if(hdp.test_data_path != ""){
                std::cout<<"\t working on predictions."<<std::endl;
                //compute (hdp.lda_alpha, hdp.lda_beta)
                 hdp.hdp_to_lda();
                 std::cout<<"lda_alpha size:"<< hdp.lda_alpha.rows()<<std::endl;

                    //prediction on the fixed test in folds
                    std::cout<<"\t working on fixed test data."<<std::endl;
                test_score = 0.0;
                test_score_split = 0.0;
                c_test_word_count_split = 0;

                for(int dd=0; dd<c_test.docs.size(); dd++){
                    document doc_test = c_test.docs[dd];
                    hdp.lda_e_step(doc_test, likelihood);
                    test_score += likelihood;
                    hdp.lda_e_step_split(doc_test, likelihood);
                    test_score_split += likelihood; 
                    c_test_word_count_split += hdp.count_split;

                }
                std::string line2w= convert(iter)+"\t"+convert(total_time)+"\t"
                    +convert(total_doc_count)+"\t"+ convert(test_score)+"\t"
                    +convert(c_test_word_count)+"\t"+ convert(test_score_split)
                    + "\t" + convert(c_test_word_count_split) +"\n";


                test_log_file_s<<line2w;
            }

        }

        //read another split

        if(hdp.seq_mode==0){
            if( split_doc_count > c_train.docs.size() * hdp.pass_ratio && num_train_splits > 1 ){

                //std::cout<< " Loading a new split from the training data"<<std::endl;
                split_doc_count = 0;
                cur_chosen_split = (cur_chosen_split +1 ) % num_train_splits;
                cur_train_filename =hdp.data_path+ train_filenames[cur_chosen_split];
                //std::cout<<"filename:"<<cur_train_filename<<std::endl;
                c_train.read_data( cur_train_filename);
                split_iter=0;
                
            }
        }
        if((hdp.max_iter != -1 && iter > hdp.max_iter)||( hdp.max_time != -1 && total_time > hdp.max_time)){
            break;
        }

    }
    log_file_s.close();
    std::cout<<"saving the final model and topics"<<std::endl;
    hdp.save_model(res_directory, -1); // when -1. flag the final model saving

    if(hdp.seq_mode){
        input_file.close();
    }

    //Making final predictions:
    std::cout<<"*** Making the final prediction."<<std::endl;
    if(hdp.test_data_path != ""){
        hdp.hdp_to_lda();
        std::cout<<"\t working on fixed test data"<<std::endl;
        test_score = 0.0;
        test_score_split = 0.0;
        c_test_word_count_split = 0;
        for( int dd =0; dd < c_test.docs.size(); dd++){
            document doc_test = c_test.docs[dd];
            hdp.lda_e_step(doc_test, likelihood);
            test_score += likelihood;
            hdp.lda_e_step_split(doc_test, likelihood);
            test_score_split += likelihood;
            c_test_word_count_split += hdp.count_split;
        }
        test_log_file_s<<iter<<"\t"<<total_time<<"\t"
            <<total_doc_count<<"\t"<<test_score<<"\t"
            <<c_test_word_count<<"\t"<<test_score_split<<"\t"
            <<c_test_word_count_split<<"\t"<<"\n";
        test_log_file_s.close();
    }
}
