#include <iostream>
#include <vector>
#include <random>
#include <cmath>
using namespace std;

class Network{
    private:
        int num_layers;
        vector<int> sizes;
        vector<vector<double>> biases;
        vector<vector<vector<double>>> weights;
    public:
        Network(vector<int> sizes){
            num_layers = sizes.size();
            this->sizes = sizes;

            random_device rd;
            mt19937 gen(rd());
            normal_distribution<double> dist(0.0, 1.0);

            for(int i = 1; i < num_layers; i++){
                vector<double> layer_biases;
                for(int j = 0; j < sizes[i]; j++){
                    layer_biases.push_back(dist(gen));
                }
                biases.push_back(layer_biases);
            }
            for(int i = 1; i < num_layers; i++){
                vector<vector<double>> weight_matrix(sizes[i], vector<double>(sizes[i-1], 0));
                for(int j = 0; j < sizes[i]; j++){
                    for(int k = 0; k < sizes[i-1]; k++){
                        weight_matrix[j][k] = dist(gen);
                    }
                }
                weights.push_back(weight_matrix);
            }
        }
};

vector<double> sigmoid(vector<double> z){
    vector<double> result;
    for(double val: z){
        result.push_back(1.0/(1.0 + exp(-val)));
    }
    return result;
}

int main(){
    cout << "Hello World!" << endl;
    return 0;
}