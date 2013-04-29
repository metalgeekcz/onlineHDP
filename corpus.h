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

#include <iostream>
#include <vector>
#include <Eigen/Dense>

#ifndef _CORPUS_H
#define _CORPUS_H

namespace HDP {

// the class for a single document
class document {
public:
    std::vector<int> words;
    std::vector<int> counts;
    int length; //length of the doc
    int total; // sum of word counts

    document():length(0),total(0) {};

    void init();
};

// the class for the whole corpus
class corpus {
public:
    int V; //the vocab size. size_vocab in the ptython code
    std::vector<document> docs; //all the documents in the corpus
    int D; //num_docs

    int num_docs; // number of documents in the corpus
    
    corpus():V(0),D(0){};

    //read in data--the second read data 
    void read_data(std::string filename);
    
    //python code is out side of the class
    void read_stream_data(std::ifstream & input, int Bsize);

    //python code is out side of the class
    int count_tokens(std::string filename);

    //python code is out side of the class
    void parse_line(std::string line, document & doc_now);

    void corpus_clear();
};

}

#endif
