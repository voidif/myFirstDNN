#include <iostream>
#include <vector>
#include "matrix.h"
#include <cmath>
#include <fstream>
#define NDEBUG
#define NDEBUG1
#define NDEBUG2
using namespace std;
void display(vector<vector<double> > a);
vector<vector<double> > dot(vector<vector<double> > a, vector<vector<double> > b);
vector<vector<double> > Hadamard(vector<vector<double> > a, vector<vector<double> > b);
vector<vector<double> > transposition(vector<vector<double> > a);
vector<vector<double> > creat();
vector<vector<double> > ran_creat(int row, int column);
vector<vector<double> > operator+(const vector<vector<double> > a, const vector<vector<double> > b);
vector<vector<double> > operator-(const vector<vector<double> > a, const vector<vector<double> > b);
vector<vector<double> > operator*(double a, const vector<vector<double> > b);
vector<vector<double> > readfile_data(ifstream& infile);
vector<vector<double> > readfile_label(ifstream& infile);
char readsingle(ifstream& infile);

class network{
public:
    network(vector<int> str){      //#biases:y*1 matrix. weights: y*x matrix. y:current layer node. x: previous layer node.
        for(int i=1;i<str.size(); i++){
            biases.push_back(ran_creat( str[i],1));   //need to be changed
            weights.push_back(ran_creat(str[i],str[i-1]));
        }
        for(int i=1;i<str.size();i++){
            delta_nabla_b.push_back(creat(0, str[i], 1));
            delta_nabla_w.push_back(creat(0, str[i], str[i-1]));
        }
        for(int i=1;i<str.size();i++){
            nabla_b.push_back(creat(0, str[i], 1));
            nabla_w.push_back(creat(0, str[i], str[i-1]));
        }
        //temp_b = biases;
        //temp_w = weights;
    }
    void net_display(){
        for(int i=1;i<biases.size();i++){
            display(biases[i]);
            display(temp_b[i]);
            display(weights[i]);
            display(temp_w[i]);
        }
    }
    vector<vector<double> > feedforward(vector<vector<double> > input);
    vector<vector<double> > sigmoid(vector<vector<double> > in);
    vector<vector<double> > sigmoid_prime(vector<vector<double> > in);
    vector<vector<double> > cost_derivative(vector<vector<double> > output_activations, vector<vector<double> > y);
    void SGD(vector<vector<vector<double> > > training_data, vector<vector<vector<double> > > test_data, int mini_batch_size, double eta);
    void update_mini_batch(vector<vector<vector<double> > > mini_batch, double eta);
    void backprop(vector<vector<vector<double> > > data);
    int evaluate(vector<vector<vector<double> > > test_data);
    void set_zero();

private:
    vector<vector<vector<double> > > biases;
    vector<vector<vector<double> > > weights;
    vector<vector<vector<double> > > delta_nabla_b;
    vector<vector<vector<double> > > delta_nabla_w;
    vector<vector<vector<double> > > nabla_b;
    vector<vector<vector<double> > > nabla_w;
    vector<vector<vector<double> > > temp_b;
    vector<vector<vector<double> > > temp_w;
};

int main(){
    #ifndef NDEBUG1
    vector<vector<vector<double> > > training_data;
    vector<vector<double> > data_data;
    vector<vector<double> > data_label;
    vector<double> data_1;
    vector<double> data_2;
    data_1.push_back(1);
    data_1.push_back(0);
    data_2.push_back(0);
    data_2.push_back(1);
    data_data.push_back(data_1);
    data_data.push_back(data_2);
    vector<double> label_1;
    vector<double> label_2;
    label_1.push_back(1);
    label_2.push_back(0);
    data_label.push_back(label_1);
    data_label.push_back(label_2);
    training_data.push_back(data_data);
    training_data.push_back(data_label);
    #endif // NDEBUG1


    ifstream infile_traindata("train-images.idx3-ubyte", ios::binary);
    ifstream infile_trainlabel("train-labels.idx1-ubyte", ios::binary);
    vector<vector<double> > trainimage_data(readfile_data(infile_traindata));
    vector<vector<double> > trainimage_label(readfile_label(infile_trainlabel));
    vector<vector<vector<double> > > training_data;
    training_data.push_back(trainimage_data);
    training_data.push_back(trainimage_label);

    ifstream infile_testdata("t10k-images.idx3-ubyte", ios::binary);
    ifstream infile_testlabel("t10k-labels.idx1-ubyte", ios::binary);
    vector<vector<double> > testimage_data(readfile_data(infile_testdata));
    vector<vector<double> > testimage_label(readfile_label(infile_testlabel));
    vector<vector<vector<double> > > test_data;
    test_data.push_back(testimage_data);
    test_data.push_back(testimage_label);



    int temp[3] = {784, 30, 10};
    vector<int> val(temp, temp+3);
    network Net(val);


    //Net.net_display();
    //Net.evaluate(test_data);
    int a;
    //cin>>a;
    for(int i=0; i<1; i++){
        cout<<i<<":";
        Net.SGD(training_data, test_data, 10, 3.0);
    }
    cout<<endl;
    //Net.net_display();
    cout<<"completed"<<endl;
    return 0;
}

vector<vector<double> > network::feedforward(vector<vector<double> > input){
    vector<vector<vector<double> > > a;
    a.push_back(input);
    for(int i=0;i<biases.size();i++){
        a.push_back(sigmoid(dot(weights[i], a[i]) + biases[i]));  //reload the + function
    }
    #ifndef NDEBUG
    cout<<"feedforward:"<<endl;
    display(a[a.size()-1]);
    cout<<endl;
    #endif // NDEBUG8
    return a[a.size()-1];
}

void network::SGD(vector<vector<vector<double> > > training_data, vector<vector<vector<double> > > test_data, int mini_batch_size, double eta){  //training without epochs;
    //int n=training_data[0].size();
    int n = 50000;
    vector<int> shuff(shuffle(n));
    //cout<<"shuff"<<shuff[0]<<"         "<<shuff[1]<<endl;
    for(int j=0;(n-j*mini_batch_size)>=mini_batch_size; j++){                                      //training_data: 2 matrix, size x pixel number and size x 1;
        //cout<<"begin update:"<<j<<endl;
        //cout<<j;
        vector<vector<vector<double> > > mini_batch;                             //mini_batch is the same as training_data
        vector<vector<double> > temp_data;
        vector<vector<double> > temp_label;
        for(int i=0;i<mini_batch_size;i++){
            temp_data.push_back(training_data[0][shuff[j*mini_batch_size+i]]);
            temp_label.push_back(training_data[1][shuff[j*mini_batch_size+i]]);
        }
        mini_batch.push_back(temp_data);
        mini_batch.push_back(temp_label);
        update_mini_batch(mini_batch, eta);
    }
    //cout<<endl;
    //cout<<"tr "<<evaluate(training_data)<<"/"<<training_data[0].size()<<"////"<<endl;;
    cout<<"te "<<evaluate(test_data)<<"/"<<test_data[0].size()<<endl;
    #ifndef NDEBUG
    cout<<"verify_zero:";
    verify_zero(weights[0]);
    verify_zero(weights[1]);
    cout<<"verify_zero b::";
    verify_zero(biases[0]);
    verify_zero(biases[1]);
    #endif // NDEBUG
}

void network::update_mini_batch(vector<vector<vector<double> > > mini_batch, double eta){
    set_zero();
    for(int i=0; i<mini_batch[0].size(); i++){           //calculate the delta of a single data
        vector<vector<vector<double> > > data;
        data.push_back(reshape(mini_batch[0][i], 1));
        data.push_back(vectorize(mini_batch[1][i][0]));  //vectorize
        backprop(data);
        for(int j=0; j<biases.size(); j++){    //add the result of delta b&w;
            nabla_b[j] = nabla_b[j] + delta_nabla_b[j];    //both nabla&nabla_nabla are initialized as all-zero matrix
            nabla_w[j] = nabla_w[j] + delta_nabla_w[j];
        }
        #ifndef NDEBUG1
        cout<<"nabla_b"<<endl;
        display(nabla_b[1]);
        cout<<endl;
        #endif // NDEBUG1
    }
    for(int k=0; k<biases.size(); k++){   //add the result to b&w;
        //cout<<"eta: "<<eta/mini_batch[0].size()<<endl;
        biases[k] = biases[k] - (eta/mini_batch[0].size())*nabla_b[k];
        weights[k] = weights[k] - (eta/mini_batch[0].size())*nabla_w[k];
    }

    //display(nabla_b[0]);
    //cout<<endl;
    //display(nabla_b[1]);
    #ifndef NDEBUG2
    cout<<"nabla_b"<<endl;
    display(nabla_b[1]);
    cout<<endl;
    cout<<"nabla_w"<<endl;
    //display(nabla_w[0]);
    cout<<endl;
    cout<<"w&b"<<endl;
    //net_display();
    cout<<endl;
    #endif // NDEBUG2
}

void network::backprop(vector<vector<vector<double> > > data){   //data::2 matrix: n x 1 and n x 1
    vector<vector<vector<double> > > zs;
    vector<vector<vector<double> > > activations;
    vector<vector<double> > activation(data[0]);
    vector<vector<double> > z;
    vector<vector<double> > delta;
    activations.push_back(activation);
    for(int bi=0; bi<weights.size(); bi++){
        z = dot(weights[bi], activation)+biases[bi];
        //display(dot(weights[bi], activation));
        //display(weights[bi]);
        zs.push_back(z);
        activation = sigmoid(z);
        activations.push_back(activation);
    }
    //backward pass begin
    delta = Hadamard(cost_derivative(activations[activations.size()-1], data[1]), sigmoid_prime(zs[zs.size()-1]));  //delta=(y-a)*f'(z)
    //delta = activations[activations.size()-1] - data[1];
    delta_nabla_b[nabla_b.size()-1] = delta;
    delta_nabla_w[nabla_w.size()-1] = dot(delta, transposition(activations[activations.size()-2]));

    for(int bj=biases.size()-2; bj>=0; bj--){
        delta=Hadamard(dot(transposition(weights[bj+1]), delta), sigmoid_prime(zs[bj]));
        delta_nabla_b[bj] = delta;
        delta_nabla_w[bj] = dot(delta, transposition(activations[bj]));
    }

    #ifndef NDEBUG
    cout<<"activations[-1]: "<<endl;
    display(activations[activations.size()-1]);
    cout<<endl;
    cout<<"zs[zs.size()-1]: "<<endl;
    display(zs[zs.size()-1]);
    cout<<endl;
    cout<<"sigmoid_prime(zs[zs.size()-1]): "<<endl;
    display(sigmoid_prime(zs[zs.size()-1]));
    cout<<endl;
    cout<<"cost_derivative(activations[activations.size()-1], data[1]): "<<endl;
    display(cost_derivative(activations[activations.size()-1], data[1]));
    cout<<endl;
    cout<<"delta: "<<endl;
    display(delta);
    cout<<endl;
    for(int k=1;k<biases.size();k++){
        cout<<"delta_nabla_b:"<<endl;
        display(delta_nabla_b[k]);
        cout<<"delta_nabla_w:"<<endl;
        display(delta_nabla_w[k]);
    }
    //cout<<"w"<<endl;
    //display(weights[0]);
    cout<<endl;
    #endif // NDEBUG
}

int network::evaluate(vector<vector<vector<double> > > test_data){
    int test_result=0;
    //verify(test_data[1]);
    int n = test_data[0].size();
    //n = 1000;
    for(int i=0; i<n; i++){
        double temp = devectorize(feedforward(reshape(test_data[0][i], 1)));
        //display(feedforward(reshape(test_data[0][i], 1)));
        if(temp==test_data[1][i][0])
            test_result++;
        //cout<<"evaluating:"<<test_result<<"/"<<i<<" "<<temp<<"/"<<test_data[1][i][0]<<"  ";
        //display(feedforward(reshape(test_data[0][i], 1)));
        //cout<<endl;
    }
    return test_result;
}

vector<vector<double> > network::cost_derivative(vector<vector<double> > output_activations, vector<vector<double> > y){
    return output_activations-y;
}

vector<vector<double> > network::sigmoid(vector<vector<double> > in){  //Activation Function
    vector<vector<double> > out;
    if(in[0].size()!=1)
        cout<<"error: not a N x 1 matrix"<<endl;
    for(int i=0;i<in.size();i++){
        vector<double> temp;
        temp.push_back((1.0/(1.0+exp(-in[i][0]))));
        out.push_back(temp);
    }
    return out;
}

vector<vector<double> > network::sigmoid_prime(vector<vector<double> > in){
    if(in[0].size()!=1)
        cout<<"error: not a N x 1 matrix"<<endl;
    return Hadamard(sigmoid(in), creat(1.0, in.size(), 1)-sigmoid(in));
}



vector<vector<double> > readfile_data(ifstream& infile){
    infile.seekg(0,ios::beg);
    char temp;
    vector<vector<double> > mnistdata;
    unsigned int a, count, l, w;
    for(int i=0; i<4; i++){
        temp=readsingle(infile);
        a = (a << 8) + (unsigned int)(unsigned char)temp;  //transforming char to unsigned char then to unsigned int, if direct from char to unsigned int, error would happen.//
    }
    for(int i=0; i<4; i++){
        temp=readsingle(infile);
        count = (count << 8) + (unsigned int)(unsigned char)temp;
    }
    for(int i=0; i<4; i++){
        temp=readsingle(infile);
        w = (w << 8) + (unsigned int)(unsigned char)temp;
    }
    for(int i=0; i<4; i++){
        temp=readsingle(infile);
        l = (l << 8) + (unsigned int)(unsigned char)temp;
    }
    cout << a << " " << count << " " << w << " " << l <<endl;
    for(unsigned int i=0;i<count;i++){
        vector<double> tempset;
        for(unsigned int j=0;j<l*w;j++){
            char readtemp;
            readtemp=readsingle(infile);
            double data = (double)(unsigned char)readtemp;
            data = data/255;
            tempset.push_back(data);
        }
        mnistdata.push_back(tempset);
    }
    return mnistdata;

}

vector<vector<double> > readfile_label(ifstream& infile){
    infile.seekg(0, ios::beg);
    char temp;
    vector<vector<double> > mnistlabel;
    unsigned int a, count;
    for(int i=0; i<4; i++){
        temp=readsingle(infile);
        a = (a << 8) + (unsigned int)(unsigned char)temp;  //transforming char to unsigned char then to unsigned int, if direct from char to unsigned int, error would happen.//
    }
    for(int i=0; i<4; i++){
        temp=readsingle(infile);
        count = (count << 8) + (unsigned int)(unsigned char)temp;
    }
    cout << a << " " << count << endl;
    for(unsigned int i=0;i<count; i++){
        vector<double> tempset;
        char readtemp;
        readtemp=readsingle(infile);
        double data = (double)(unsigned char)readtemp;
        tempset.push_back(data);
        mnistlabel.push_back(tempset);
    }
    cout<<mnistlabel[0][0];
    cout<<endl;

    return mnistlabel;
}

char readsingle(ifstream& infile){
    char buffer;
    infile.read(&buffer, 1);
    return buffer;
}

void network::set_zero(){
    for(int i=0;i<nabla_b.size();i++){
        for(int j=0;j<nabla_b[i].size();j++){
            for(int k=0; k<nabla_b[i][j].size();k++){
                nabla_b[i][j][k]=0;
            }
        }
    }
    for(int i=0;i<nabla_w.size();i++){
        for(int j=0;j<nabla_w[i].size();j++){
            for(int k=0; k<nabla_w[i][j].size();k++){
                nabla_w[i][j][k]=0;
            }
        }
    }
}
