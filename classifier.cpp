#include "classifier.h"
#include <math.h>
#include <string>
#include <vector>
#include <iostream>

using Eigen::ArrayXd;
using std::string;
using std::vector;

// Initializes GNB
GNB::GNB() {}

GNB::~GNB() {}

void GNB::train(const vector<vector<double>> &data, 
                const vector<string> &labels) {
  /**
   * Trains the classifier with N data points and labels.
   * @param data - array of N observations
   *   - Each observation is a tuple with 4 values: s, d, s_dot and d_dot.
   *   - Example : [[3.5, 0.1, 5.9, -0.02],
   *                [8.0, -0.3, 3.0, 2.2],
   *                 ...
   *                ]
   * @param labels - array of N labels
   *   - Each label is one of "left", "keep", or "right".
   *
   * Implement the training function for  classifier.
   * i.e. find mean and stdev of features for each label
   */
   
   n_features = data[0].size();
   n_examples = data.size();
   n_labels = 3;

   // group by labels and features
   vector<vector<double>> sums (n_labels, vector<double> (n_features, 0));
   vector<double> n_labels_i (n_labels,0);
   int lab_i;
   for (int i = 0; i < n_examples; i++) {
        if (labels[i].compare("left") == 0) {
            lab_i = 0;
        } else if (labels[i].compare("keep") == 0) {
            lab_i = 1;
        } else if (labels[i].compare("right") == 0) {
            lab_i = 2;
        } 
        n_labels_i[lab_i] += 1; 
        for (int j=0; j < n_features; j++) {
            sums[lab_i][j] += data[i][j];
       }
    }

    means = sums;   
    priors = n_labels_i;
    for (int i = 0; i < n_labels; i++) {
        priors[i] /= n_examples;
        for (int j=0; j < n_features; j++) {
            means[i][j] /= n_labels_i[i];
        }
    }
    
    std::cout << "means:" << std::endl;
    for (auto row : means) {
        for (auto item : row) {
            std::cout << item << "   ";
        }
        std::cout << " " << std::endl;
    };
    
    vector<vector<double>> diffs (n_labels, vector<double> (n_features, 0.0));
    for (int i = 0; i < n_examples; i++) {
        if (labels[i].compare("left") == 0) {
            lab_i = 0;
        } else if (labels[i].compare("keep") == 0) {
            lab_i = 1;
        } else if (labels[i].compare("right") == 0) {
            lab_i = 2;
        }  
        for (int j=0; j < n_features; j++) {
            diffs[lab_i][j] += pow(means[lab_i][j] - data[i][j], 2.0);
       }
    }

    stdevs = diffs;
    for (int i = 0; i < n_labels; i++) {
        for (int j=0; j < n_features; j++) {
            stdevs[i][j] = pow(diffs[i][j] / n_labels_i[i], 0.5) ;
        }
    }    
    std::cout << "stdevs:" << std::endl;
    for (auto row : stdevs) {
        for (auto item : row) {
            std::cout << item << "   ";
        }
        std::cout << " " << std::endl;
    };
}

string GNB::predict(const vector<double> &sample) {
  /**
   * Once trained, this method is called and expected to return 
   *   a predicted behavior for the given observation.
   * @param observation - a 4 tuple with s, d, s_dot, d_dot.
   *   - Example: [3.5, 0.1, 8.5, -0.2]
   * @output A label representing the best guess of the classifier. Can
   *   be one of "left", "keep" or "right".
   */
  
  // Compute the conditional probabilities for each feature/label combination
  // p(x=v|c) = 1/(2*pi*sig^2)  * exp( - (v-mu)^2 / (2*sig^2) )

  // Use the conditional probabilities in a Naive Bayes classifier
  // y = argmax (p_Ck) * PI( p(x_i=v_i|c_k) )
  vector<double> probs (n_labels, 1.0);
  for (int i=0; i < n_labels; i++) {
    for (int j=0; j<n_features; j++) {
      double sig = stdevs[i][j];
      double mu = means[i][j];
      double v = sample[j];
      double var = pow(sig,2.0);
      double p_i = 1.0/(2.0*M_PI*var)  * exp( - pow(v-mu, 2.0) / (2.0*var) );
      // std::cout << p_i << "  ";
      probs[i] *= p_i;
    }
    //std::cout << " -> " << probs[i] << std::endl;
  }
  int ans = std::distance(probs.begin(), std::max_element(probs.begin(), probs.end()));
  // std::cout << " ----  " << "  " << ans << std::endl;  
  return this -> possible_labels[ans];
}