// ------------------------------------------------------------------------
// Copyright (C) 2015 Che-Wei Huang <cheweihuang@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// ------------------------------------------------------------------------

#ifndef LDA_H_
#define LDA_H_

#include <iostream>
#include <math.h>
#include <time.h>
#include <limits>
#include <algorithm>
#include <map>
#include <iomanip>
#include <iterator>

#include <omp.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>


namespace lda {

typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::Triplet<int> Tri;
typedef Eigen::SparseMatrix<double, Eigen::RowMajor> RSpMat;
typedef Eigen::Matrix<double,
                      Eigen::Dynamic,
                      Eigen::Dynamic,
                      Eigen::ColMajor> CMatrixXd;
typedef Eigen::Matrix<double,
                      Eigen::Dynamic,
                      Eigen::Dynamic,
                      Eigen::RowMajor> RMatrixXd;
typedef Eigen::Matrix<int,
                      Eigen::Dynamic,
                      Eigen::Dynamic,
                      Eigen::RowMajor> RMatrixXi;
struct Corpus
{ 
  bool bEmpty;
  SpMat spmCorpus; // W x D sparse matrix for the bag of words
  std::vector<int> vecDocHeader; // {D, W, N }
  Corpus () {bEmpty = true;}
};

class LDA {
  public:
    explicit LDA (const int iNIters, // number of iterations
                  const int iK, // number of Topics
                  const int iD, // number of documents
                  const int iW, // size of vocabulary
                  const int iN, // number of observations(words)
                  const int iMiniBatchSize = 100, // mini-batch size
                  const int iBurnInPerDoc = 3, // number of burn-in phases per doc

                  /* Dirichlet prior parameters */
                  const double dEta = 0.01, 
                  const double dAlpha = 0.1, 

                  /* step size parameters */
                  const double sw = 1.0,
                  const double sd = 10.0,
                  const double tauw = 10.0,
                  const double taud = 1000.0,
                  const double kappa = 0.9
                  )

    : m_iMaxNIters(iNIters), m_iK(iK), m_iD(iD),
    m_iW(iW), m_iN(iN), m_iMiniBatchSize(iMiniBatchSize),
    m_iBurnInPerDoc(iBurnInPerDoc),
    m_iMiniBatchesPerCorpus(m_iD/m_iMiniBatchSize),
    m_dEta(dEta), m_dAlpha(dAlpha), m_dSW(sw), m_dSD(sd), 
    m_dTauW(tauw), m_dTauD(taud), m_dKappa(kappa)
    {}

    ~LDA() {}

    bool fit(const SpMat&);// fit data to topic model
    const CMatrixXd& Phi() {return m_matPhi;}
    const CMatrixXd& Theta() {return m_matTheta;}

  private:
    int m_iMaxNIters, m_iK, m_iD, m_iW, m_iN,
        m_iMiniBatchSize, m_iBurnInPerDoc, m_iMiniBatchesPerCorpus,
        m_iRhoThetaT, m_iRhoPhiT;
    long int m_iC; // total word counts in the corpus
    double m_dEta, m_dAlpha, m_dSW, m_dSD, m_dTauW, m_dTauD, m_dKappa;

    Eigen::VectorXd m_vecC, // D-vector: word counts for each document
                    m_vecNZ; // CVB0 statistics
    CMatrixXd m_matNPhi, m_matNTheta, // CVB0 statistics
              m_matPhi, m_matTheta; // model distributions
    bool CVB0StatsInit(const SpMat&); // initialization of CVB0 statistics
    bool SCVB0(const SpMat&); // inference algorithm
    bool BurnIn(const SpMat&, const int); // burn-in phase
    bool CalPosteriorDists(); // compute posterior distributions
    double CorpusLogLikelihood(const SpMat&); // compute perplexity
};

} // namespace lda
#endif // LDA_H_
