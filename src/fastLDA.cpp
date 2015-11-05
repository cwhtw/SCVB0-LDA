// ------------------------------------------------------------------------
//  Copyright (C) 2015 Che-Wei Huang <cheweihuang@gmail.com>
//
//  This Source Code Form is subject to the terms of the Mozilla
//  Public License v. 2.0. If a copy of the MPL was not distributed
//  with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// ------------------------------------------------------------------------
//  Parallel C++/openMp implementation for the SCVB0 algorithm
//  presented in the paper below.
//
//  @inproceedings{Foulds:2013:SCV:2487575.2487697,
//   author = {Foulds, James and Boyles, Levi and DuBois, 
//   Christopher and Smyth, Padhraic and Welling, Max},
//   title = {Stochastic Collapsed Variational Bayesian Inference 
//            for Latent Dirichlet Allocation},
//  } 
//  ------------------------------------------------------------------------
//  Usage: ./fastLDA docword.txt iterations numOfTopics
//  Inputs:
//    docword.txt: bag-of-word file of format as that in UCI bag-of-word files
//    iterations: number of iterations 
//    numOfTopics: number of topics
//
//  Outputs:
//    topics.txt: 
//      Top 100 most likely words per topic per line, 
//      in a descending order of likellihood
//    doctopic.txt: 
//      Topic mixture distribution per line per document,
//      in an ascending order of document IDs.

#include <fstream>
#include <sstream>

#include "LDA.h"
using namespace lda;

const Corpus&
loadCorpus(const std::string &strPathDocWord) {
  
  static Corpus inCorpus;
  std::vector<Tri> vecTris;

	std::ifstream inFile(strPathDocWord.c_str());
  if (!inFile) {
    std::cerr << "File " << strPathDocWord << " not found!" << std::endl;
    return inCorpus;
  }

  int iWordCountCorpus = 0;
  std::string strTmpLine;
  while (std::getline(inFile, strTmpLine)) {
    if (strTmpLine.empty()) continue;
    iWordCountCorpus += 1;
    if ((int)inCorpus.vecDocHeader.size() < 3) {
          inCorpus.vecDocHeader.push_back(atoi(strTmpLine.c_str()));
    } else {
      std::istringstream tokenizer(strTmpLine);
      int iDocID, iWordID, iCount;
      tokenizer >> iDocID >> iWordID >> iCount;
      vecTris.push_back(Tri(iWordID-1, iDocID-1, iCount));
      if ((int)vecTris.size()%100000 == 0) {
        std::cerr << "Have loaded " 
                  << vecTris.size() 
                  << " words" << std::endl;
      }
    }
  }
  inFile.close();

  // convert into a sparse matrix
  inCorpus.spmCorpus.resize(inCorpus.vecDocHeader[1], 
                            inCorpus.vecDocHeader[0]); // W x D sparse mat
  inCorpus.spmCorpus.setFromTriplets(vecTris.begin(), vecTris.end());
  inCorpus.spmCorpus.makeCompressed();
  
  inCorpus.bEmpty = false;
  return inCorpus;
}

int
main(int argc, char *argv[]) {

  std::ios::sync_with_stdio(false);
  try {
    if (argc < 4) {
      std::string usage = 
        "Topic Modeling Using Latent Dirichlet Allocation "
        "(SCVB0 inference algorithm)\n"
        "Usage: ./fastLDA docword.txt iterations NumOfTopics";
      throw usage;
    }
  } catch (const std::string usage) {
    std::cerr << usage << std::endl;
    return -1;
  }

  const std::string strPathDocWord = argv[1]; 
  const int iNIters = atoi(argv[2]), iNTopics = atoi(argv[3]);
  std::cerr <<  "Executing the command: " 
            << argv[0] << " " <<  strPathDocWord << " " 
            << iNIters << " " << iNTopics << std::endl;

  // loading corpus
  const double dStartLoad = omp_get_wtime();
  const Corpus &inCorpus = loadCorpus(strPathDocWord);
  const double dEndLoad = omp_get_wtime();
  if (inCorpus.bEmpty) {return -1;}
  std::cerr << "Loading took: " 
            << dEndLoad - dStartLoad 
            << " seconds" << std::endl;
  // end of loading corpus

  // fitting topic model
  LDA lda(iNIters, iNTopics, 
          inCorpus.vecDocHeader[0],   // D
          inCorpus.vecDocHeader[1],   // W
          inCorpus.vecDocHeader[2]);  // N
  const double dStartFitLDA = omp_get_wtime();
  const bool success = lda.fit(inCorpus.spmCorpus);//, inCorpus.vecC);
  const double dEndFitLDA = omp_get_wtime();
  if (!success) {
    std::cerr << "Failed to fit the corpus into a topic model!\n";
    return -1;
  } else {
    std::cerr << "Fitting the topic model took: " 
              << dEndFitLDA - dStartFitLDA 
              << " seconds" << std::endl;
  }
  // end of fitting topic model

  // write to output files
  const CMatrixXd &matPhi = lda.Phi(), &matTheta = lda.Theta();
  // topics.txt
  const int iNTopWords = 100;
  const std::string strTopicsFileName = "topics.txt";
  std::vector<int> vecWordID(inCorpus.vecDocHeader[1]);
  for (int w=0; w<inCorpus.vecDocHeader[1]; w++) {
    vecWordID[w] = w;
  }
  RMatrixXd matTopicsTopWordsProb(iNTopics, iNTopWords);
  RMatrixXi matTopicsTopWordsID(iNTopics, iNTopWords);
  std::vector<double> vecProb(inCorpus.vecDocHeader[1]);
  for (int k=0; k<matPhi.cols(); k++) {
    Eigen::Map< CMatrixXd > (vecProb.data(), matPhi.rows(), 1) = matPhi.col(k);
    //std::cout << vecProb[0] << std::endl;
    std::map<double, int> mapProbID;
    std::transform(vecProb.begin(), vecProb.end(), vecWordID.begin(),
                    std::inserter(mapProbID, mapProbID.begin()),
                    &std::make_pair<double, int>);
    int w = 0;
    for (std::map<double,int>::reverse_iterator rit=mapProbID.rbegin();
        rit != mapProbID.rend(), w<iNTopWords; ++rit, ++w) {
      matTopicsTopWordsProb(k,w) = rit->first;
      matTopicsTopWordsID(k,w) = rit->second;
    }
  }
  std::ofstream topics_fout(strTopicsFileName.c_str());
  if (topics_fout.is_open()) {
    for (int k=0; k<matPhi.cols(); k++) {
      for (int w=0; w<iNTopWords; w++) {
        topics_fout << matTopicsTopWordsID(k,w)+1 << ":"
                    << std::setprecision(6)
                    << matTopicsTopWordsProb(k,w); 
        if (w==iNTopWords-1) 
          topics_fout << "\n";
        else
          topics_fout << ", ";
      }
    }
  }
  topics_fout.close();

  // doctopic.txt
  const std::string strDocTopicsFileName = "doctopic.txt";
  std::ofstream doctopic_fout(strDocTopicsFileName.c_str());
  if (doctopic_fout.is_open()) {
    for (int d=0; d<matTheta.rows(); d++) {
      for (int k=0; k<matTheta.cols(); k++) {
        doctopic_fout << std::setprecision(6) 
                      << matTheta(d,k);
        if (k==matTheta.cols()-1)
          doctopic_fout << "\n";
        else
          doctopic_fout << ", ";
      }
    }
  }
  doctopic_fout.close();

  return 0;
}
