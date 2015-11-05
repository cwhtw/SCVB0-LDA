// ------------------------------------------------------------------------
// Copyright (C) 2015 Che-Wei Huang <cheweihuang@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// ------------------------------------------------------------------------

#include "LDA.h"

namespace lda {

bool
LDA::fit(const SpMat &spmCorpus) {
  std::srand((unsigned int) time(0));

  m_vecC.resize(spmCorpus.outerSize());
  for (int d=0; d<spmCorpus.outerSize(); d++) {
    m_vecC(d) = spmCorpus.innerVector(d).sum();
  }
  m_iC = m_vecC.sum();

  CVB0StatsInit(spmCorpus);

  const bool success = SCVB0(spmCorpus);
  if (!success) {
    std::cerr << "In LDA::fit, SCVB0 inference algorithm failed!" << std::endl;
    return false;
  }

  return true;
}

bool
LDA::CVB0StatsInit(const SpMat &spmCorpus) {

  m_matNPhi = Eigen::MatrixXd::Zero(m_iW, m_iK);
  m_matNTheta = Eigen::MatrixXd::Zero(m_iD, m_iK);
  m_vecNZ = Eigen::VectorXd::Zero(m_iK);
  for (int d=0; d<spmCorpus.outerSize(); d++) {
    for (SpMat::InnerIterator wordit(spmCorpus, d); wordit; ++wordit) {
      Eigen::VectorXd vecUN = 
        0.5 * (Eigen::VectorXd::Random(m_iK)
        + Eigen::VectorXd::Constant(m_iK, 1.0)),
        vecMM = (vecUN / vecUN.sum()) * wordit.value();
      m_matNPhi.row(wordit.row()) += vecMM;
      m_matNTheta.row(wordit.col()) += vecMM;
    }
  }
  m_vecNZ = m_matNPhi.colwise().sum();
  
  return true;
}

bool
LDA::BurnIn(const SpMat &spmCorpus,
            const int iDocID) {

  for (int burnin=0; burnin<m_iBurnInPerDoc; burnin++) {
    double dRhoTheta = m_dSW / pow(m_dTauW + burnin, m_dKappa);
    for (SpMat::InnerIterator wordit(spmCorpus, iDocID); wordit; ++wordit) {
      Eigen::VectorXd matGammaUN = // un-normalized Eq.(5)
        (m_matNPhi.row(wordit.row()).array() + m_dEta) 
        * (m_matNTheta.row(iDocID).array() + m_dAlpha) 
        / (m_vecNZ.transpose().array() + m_dEta*m_iW);
      Eigen::VectorXd matGamma = matGammaUN.array() / matGammaUN.sum();
      // update NTheta Eq.(9)
      Eigen::VectorXd vecTmp =                                             
        pow(1.0 - dRhoTheta, wordit.value()) * m_matNTheta.row(iDocID)
        + (1.0 - pow(1.0 - dRhoTheta, wordit.value()))
        * m_vecC(iDocID) * matGamma.transpose();
      m_matNTheta.row(iDocID) = vecTmp;
    } // end of wordit
  } // end of burnin
  return true;
}

bool 
LDA::SCVB0(const SpMat &spmCorpus) {

  // use max number of processors
  omp_set_num_threads(omp_get_max_threads());
  
  double dOldPerp = 0.0;
  for (int iter=0; iter<m_iMaxNIters; iter++) {
    const double dIterBeginTime = omp_get_wtime();
    double dRhoPhi = m_dSD / pow(m_dTauD+iter, m_dKappa);

    #pragma omp parallel for // parallize over mini-batches
    for (int batch=0; batch<m_iMiniBatchesPerCorpus; batch++) {
      Eigen::MatrixXd matNPhi_hat = Eigen::MatrixXd::Zero(m_iW, m_iK);
      Eigen::VectorXd vecNZ_hat = Eigen::VectorXd::Zero(m_iK);

      for (int d=batch*m_iMiniBatchSize; d<(batch+1)*m_iMiniBatchSize; d++) {
        if (m_vecC(d) == 0) continue;
        BurnIn(spmCorpus,d);
        const double dRhoTheta = m_dSW / pow(m_dTauW + m_vecC(d), m_dKappa);

        // update
        for (SpMat::InnerIterator wordit(spmCorpus, d); wordit; ++wordit) {
          Eigen::VectorXd matGammaUN = // un-normalized Eq.(5)
            (m_matNPhi.row(wordit.row()).array() + m_dEta) \
            * (m_matNTheta.row(d).array() + m_dAlpha) \
            / (m_vecNZ.transpose().array() + m_dEta*m_iW);
          Eigen::VectorXd matGamma = matGammaUN.array() / matGammaUN.sum();

          // update NTheta Eq.(9)
          Eigen::VectorXd vecTmp =
            pow(1.0 - dRhoTheta, wordit.value()) * m_matNTheta.row(d)
            + (1.0 - pow(1.0 - dRhoTheta, wordit.value()))
            * m_vecC(d) * matGamma.transpose();
          m_matNTheta.row(d) = vecTmp;
          
          // update minibatch counts NPhi_hat, NZ_hat
          matNPhi_hat.row(wordit.row()) += matGamma.transpose();
          vecNZ_hat += matGamma.transpose();

        } // end of wordit 
      } // end of d

      // update NPhi, NZ Eq.(7-8)
      #pragma omp critical
      {
        Eigen::MatrixXd matTmp = (1.0 - dRhoPhi) * m_matNPhi 
          + dRhoPhi * (1.0 * m_iC / m_iMiniBatchSize) * matNPhi_hat;
        Eigen::VectorXd vecTmp = (1.0 - dRhoPhi) * m_vecNZ 
          + dRhoPhi * (1.0 * m_iC / m_iMiniBatchSize) * vecNZ_hat;
        m_matNPhi = matTmp;
        m_vecNZ = vecTmp;
      }

    } // end of batch

    const double dIterEndTime = omp_get_wtime();

    CalPosteriorDists();
    const double dPerplexity = exp2(-CorpusLogLikelihood(spmCorpus)/m_iC);

    std::cerr << "Iteration " << iter << " took "
              << std::setprecision(7)
              << dIterEndTime - dIterBeginTime
              << " seconds (perplexity: "
              << std::setprecision(6)
              << dPerplexity 
              << ")" << std::endl;

    if ((iter>=1) && ((dOldPerp-dPerplexity)/dOldPerp) < 0.005) {break;}
    dOldPerp = dPerplexity;

  } // end of iter

  return true;
}

bool
LDA::CalPosteriorDists() {

  // normalize Phi matrix
  Eigen::MatrixXd matTmp = m_matNPhi.array() + m_dEta;
  m_matPhi = matTmp.array().rowwise() / matTmp.array().colwise().sum();
  // notmalize Theta matrix
  matTmp = m_matNTheta.array() + m_dAlpha;
  m_matTheta = matTmp.array().colwise() / matTmp.array().rowwise().sum();
  return true;
}

double
LDA::CorpusLogLikelihood(const SpMat& spmCorpus) {

  Eigen::VectorXd vecLL = Eigen::VectorXd::Zero(spmCorpus.outerSize());

  #pragma omp parallel for
  for (int d=0; d<spmCorpus.outerSize(); d++) {
    double dLL = 0.0;
    for (SpMat::InnerIterator wordit(spmCorpus, d); wordit; ++wordit) {
      const double pW = m_matPhi.row(wordit.row()).dot(m_matTheta.row(d)),
                   log2pW = log2(pW) * wordit.value();
      dLL += log2pW;
    }
    vecLL(d) = dLL;
  }
  const double LL = vecLL.sum();
  return LL;
}

} // namespace lda
