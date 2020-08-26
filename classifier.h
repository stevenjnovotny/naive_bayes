#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include <string>
#include <vector>
#include "Dense"

using Eigen::ArrayXd;
using std::string;
using std::vector;

class GNB {
 public:
  /**
   * Constructor
   */
  GNB();
  vector<vector<double>> means; // mu[label, feature]
  vector<vector<double>> stdevs;  // std[label, feature]
  vector<double> priors; // p(c_k)
  
  int n_features;
  int n_examples;
  int n_labels;

  /**
   * Destructor
   */
  virtual ~GNB();

  /**
   * Train classifier
   */
  void train(const vector<vector<double>> &data, 
             const vector<string> &labels);

  /**
   * Predict with trained classifier
   */
  string predict(const vector<double> &sample);
  
  vector<string> possible_labels = {"left","keep","right"};
};

#endif  // CLASSIFIER_H