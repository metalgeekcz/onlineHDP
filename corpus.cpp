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
 
#include "corpus.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <vector>
#include <cstring>

namespace HDP {

void corpus::read_data(std::string filename){

    corpus_clear();

    std::ifstream input(filename.c_str());

    std::string line;

    while(getline(input, line)){
        document doc_now;
        int max_word=0; // the max word id of this document
        std::vector<std::string> tokens;
        std::istringstream iss(line);
        int tt=0;
        int doc_id;
        do{
            std::string sub;
            iss >> sub; // get a token with space as delim
            if(tt==0){
                doc_id = atoi(sub.c_str());
            }else{
                std::string token; //get every number
                std::stringstream stream(sub);
                int ii=0;
                while(getline(stream, token, ':')){
                    if(ii==0){
                        doc_now.words.push_back(atoi(token.c_str()));

                        if(atoi(token.c_str()) > max_word){
                            max_word=atoi(token.c_str());
                        }
                    }else{
                        doc_now.counts.push_back(atoi(token.c_str()));
                        doc_now.total += atoi(token.c_str());
                        doc_now.length ++;
                    }
                    ii++;
                }
            }

            tt++;
        }while(iss);
        
        
        assert( doc_now.words.size() == doc_now.counts.size() );

        //count the number of docs
        D++;
        //push back the doc to the corpus
        docs.push_back(doc_now);

        //update V
        if( max_word > V){
            V=max_word;
        }
    }

    num_docs = docs.size();
}
 
void corpus::read_stream_data(std::ifstream & input, int Bsize){
    
    //clear the old data
    docs.clear();
    std::string line;

    for ( int dd=0; dd<Bsize; dd++){

        getline(input, line);
        if(input.eof()){
            break;
        }
        document doc_now;

        std::vector<std::string> tokens;
        std::istringstream iss(line);
        int tt=0;
        int doc_id;
        do{
            std::string sub;
            iss >> sub; // get a token with space as delim
            if (tt == 0){
                doc_id = atoi(sub.c_str());
                //cout<<"doc_id"<<doc_id<<endl;
            }else{
                std::string token; //get every number
                std::stringstream stream(sub);
                int ii=0;
                while(getline(stream, token, ':')){
                    if(ii==0){
                        doc_now.words.push_back(atoi(token.c_str()));
                    }else{
                        doc_now.counts.push_back(atoi(token.c_str()));
                        doc_now.total += atoi(token.c_str());
                        doc_now.length ++;
                    }
                    ii++;
                }
            }
            tt++;
        }while(iss);
        
        tt++;
        assert( doc_now.words.size() == doc_now.counts.size() );

        //count the number of docs
        D++;
        //push back the doc to the corpus
        docs.push_back(doc_now);
    }

    num_docs = docs.size();
}

void corpus::corpus_clear(){
    docs.clear();
    D=0;
    num_docs=0;
}

}
