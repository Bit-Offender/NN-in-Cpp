#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <string>
#include <fstream>
#include <sstream>
#include <Eigen/Dense>
#include <algorithm>

using namespace std;
using namespace Eigen;

VectorXd sigmoid(VectorXd z){
    return (1.0/(1.0 + (-z).array().exp()));
}

VectorXd sigmoid_prime(VectorXd z){
    VectorXd s = sigmoid(z);
    return (s.array()*(1 - s.array()));
}

vector<pair<VectorXd, VectorXd>> load_test_data(){
    ifstream file("./mnist_test.csv/mnist_test.csv");

    vector<pair<VectorXd, VectorXd>> csvData;

    if(!file.is_open()){
        cerr << "Could not open test csv!" << endl;
        return csvData;
    }

    string line;
    getline(file, line); //skip header line

    csvData.reserve(10000);

    while(getline(file, line)){
        stringstream ss(line);
        string cell;

        VectorXd target(10), data(784);
        target.setZero();
        data.setZero();

        getline(ss, cell, ',');
        target(stoi(cell)) = 1.0;

        for(int i = 0; i < 784 && getline(ss, cell, ','); i++){
            data(i) = stod(cell)/255.0;
        }

        csvData.emplace_back(move(data), move(target));
    }

    return csvData;
}

vector<pair<VectorXd, VectorXd>> load_training_data(){
    ifstream file("./mnist_train.csv/mnist_train.csv");

    vector<pair<VectorXd, VectorXd>> csvData;

    if(!file.is_open()){
        cerr << "Could not open training csv!" << endl;
        return csvData;
    }
    
    string line;
    getline(file, line); //skip header line

    csvData.reserve(60000);

    while(getline(file, line)){
        stringstream ss(line);
        string cell;

        VectorXd target(10), data(784);
        target.setZero();
        data.setZero();

        getline(ss, cell, ',');
        target(stoi(cell)) = 1.0;

        for(int i = 0; i < 784 && getline(ss, cell, ','); i++){
            data(i) = stod(cell)/255.0; //normalize pixels to 0 to 1
        }

        csvData.emplace_back(move(data), move(target));
    }

    return csvData;
}

class Network{
    private:
        int num_layers;
        vector<int> sizes;
        vector<VectorXd> biases;
        vector<MatrixXd> weights;
    public:
        Network(vector<int> sizes){
            num_layers = sizes.size();
            this->sizes = sizes;

            random_device rd;
            mt19937 gen(rd());
            normal_distribution<double> dist(0.0, 1.0);

            for(int i = 1; i < num_layers; i++){
                VectorXd layer_biases(sizes[i]);
                for(int j = 0; j < sizes[i]; j++){
                    layer_biases(j) = dist(gen);
                }
                biases.push_back(layer_biases);
            }
            for(int i = 1; i < num_layers; i++){
                MatrixXd weight_matrix(sizes[i], sizes[i-1]);
                for(int j = 0; j < sizes[i]; j++){
                    for(int k = 0; k < sizes[i-1]; k++){
                        weight_matrix(j, k) = dist(gen);
                    }
                }
                weights.push_back(weight_matrix);
            }
        }

        VectorXd cost_derivative(const VectorXd& output_activations, const VectorXd& y){
            return (output_activations - y);
        }

        pair<vector<VectorXd>, vector<MatrixXd>> backprop(const VectorXd& x, const VectorXd& y){
            vector<VectorXd> nabla_b = biases;
            vector<MatrixXd> nabla_w = weights;
            for(size_t i = 0; i < weights.size(); i++){
                nabla_b[i].setZero();
                nabla_w[i].setZero();
            }

            VectorXd activation = x;
            vector<VectorXd> activations = {x};
            vector<VectorXd> zs;

            for(size_t i = 0; i < weights.size(); i++){
                VectorXd z = (weights[i] * activation + biases[i]);
                zs.push_back(z);
                activation = sigmoid(z);
                activations.push_back(activation);
            }

            VectorXd delta = (cost_derivative(activations[activations.size()-1], y).array() * sigmoid_prime(zs[zs.size()-1]).array()).matrix();

            nabla_b[nabla_b.size()-1] = delta;
            nabla_w[nabla_w.size()-1] = delta * activations[activations.size()-2].transpose();

            for(int i = 2; i < num_layers; i++){
                VectorXd z = zs[zs.size()-i];
                VectorXd sp = sigmoid_prime(z);
                delta = (weights[weights.size()-i+1].transpose() * delta).array() * sp.array();
                nabla_b[nabla_b.size()-i] = delta;
                nabla_w[nabla_w.size()-i] = delta * activations[activations.size()-i-1].transpose();
            }
            
            return {nabla_b, nabla_w};
        } 

        void update_mini_batch(const vector<pair<VectorXd, VectorXd>>& mini_batch, double eeta){
            vector<VectorXd> nabla_b = biases;
            vector<MatrixXd> nabla_w = weights;
            for(size_t i = 0; i < weights.size(); i++){
                nabla_b[i].setZero();
                nabla_w[i].setZero();
            }
            for(size_t j = 0; j < mini_batch.size(); j++){

                // pair<vector<VectorXd>, vector<MatrixXd>> result = backprop(x, y);
                // vector<VectorXd> delta_nabla_b = result.first;
                // vector<MatrixXd> delta_nabla_w = result.second;
                // SAME AS BELOW:
                
                auto [delta_nabla_b, delta_nabla_w] = backprop(mini_batch[j].first, mini_batch[j].second);
                for(size_t l = 0; l < weights.size(); l++){
                    nabla_b[l] += delta_nabla_b[l];
                    nabla_w[l] += delta_nabla_w[l];
                }
            }

            for(size_t l = 0; l < weights.size(); l++){
                weights[l] -= (eeta/mini_batch.size()) * nabla_w[l]; 
                biases[l] -= (eeta/mini_batch.size()) * nabla_b[l]; 
            }
        }

        VectorXd feedforward(VectorXd a){
            for(size_t i = 0; i < weights.size(); i++){
                a = sigmoid(weights[i] * a + biases[i]);
            }
            return a;
        }

        int evaluate(const vector<pair<VectorXd, VectorXd>>& test_data){
            int correct = 0;
            for(size_t i = 0; i < test_data.size(); i++){
                int predicted, actual;
                feedforward(test_data[i].first).maxCoeff(&predicted);
                test_data[i].second.maxCoeff(&actual);
                correct += (predicted == actual);
            }
            return correct;
        }

        void StochasticGradientDescent(
            vector<pair<VectorXd, VectorXd>>& training_data, 
            int epochs, 
            int mini_batch_size, 
            double eeta, 
            const vector<pair<VectorXd, VectorXd>>& test_data={})
        {
            random_device rd;
            mt19937 gen(rd());

            int n_test = test_data.size();

            int n = training_data.size();

            for(int i = 0; i < epochs; i++){

                shuffle(training_data.begin(), training_data.end(), gen);

                vector<vector<pair<VectorXd, VectorXd>>> mini_batches;

                for(int k = 0; k < n; k += mini_batch_size){
                    int end = min(k+mini_batch_size, n);
                    mini_batches.emplace_back(training_data.begin() + k, training_data.begin() + end);  
                }

                for(const auto& mini_batch: mini_batches){
                    update_mini_batch(mini_batch, eeta);
                }
                if(n_test > 0){
                    cout << "Epoch " << i << ": " << evaluate(test_data) << " / " << n_test << endl;
                } else {
                    cout << "Epoch " << i << " complete" << endl;
                }
            }
        }

};

int main(){
    
    auto training_data = load_training_data();
    auto test_data = load_test_data();

    if(training_data.empty() || test_data.empty()) return 1;

    Network net({784, 30, 10});

    net.StochasticGradientDescent(training_data, 30, 10, 3.0, test_data);

    return 0;
}