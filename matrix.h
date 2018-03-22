#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <stdlib.h>
#include <cmath>
#include <algorithm>
using namespace std;
void display(vector<vector<double> > a);
double gaussrand();
vector<vector<double> > dot(vector<vector<double> > a, vector<vector<double> > b);
vector<vector<double> > Hadamard(vector<vector<double> > a, vector<vector<double> > b);
vector<vector<double> > transposition(vector<vector<double> > a);
vector<vector<double> > creat();
vector<vector<double> > creat(double val, int row, int column);
vector<vector<double> > ran_creat(int row, int column);
vector<vector<double> > reshape(vector<double> a, int column);
vector<vector<double> > vectorize(int a);
vector<vector<double> > vectorize_single(int a);
int devectorize(vector<vector<double> > a);
void verify(vector<vector<double> > a);
vector<vector<double> > operator+(const vector<vector<double> > a, const vector<vector<double> > b);
vector<vector<double> > operator-(const vector<vector<double> > a, const vector<vector<double> > b);
vector<vector<double> > operator*(double a, const vector<vector<double> > b);
void verify_zero(vector<vector<double> > a);
vector<int> shuffle(int n);


//int main(){vector<vector<vector<double> > > data
//    int c[2][2]={{1,0},{0,1}};
//    vector<vector<double> > a(creat());
//    display(a);
//    return 0;
//}


double gaussrand()
{
    //srand((unsigned)time(NULL));
    static double V1, V2, S;
    static int phase = 0;
    double X;

    if ( phase == 0 ) {
        do {
            double U1 = (double)rand() / RAND_MAX;
            double U2 = (double)rand() / RAND_MAX;

            V1 = 2 * U1 - 1;
            V2 = 2 * U2 - 1;
            S = V1 * V1 + V2 * V2;
        } while(S >= 1 || S == 0);

        X = V1 * sqrt(-2 * log(S) / S);
    } else
        X = V2 * sqrt(-2 * log(S) / S);

    phase = 1 - phase;

    return X;
}

vector<vector<double> > ran_creat(int row, int column){  //create random matrix
    vector<vector<double> > resul;
    //srand((unsigned)time(NULL)*(unsigned)(row-column));
    double val;
    for(int crow=0;crow<row;crow++){
        vector<double> temp;
        for(int ccolumn=0;ccolumn<column;ccolumn++){
            val =  gaussrand();
            //val=rand()/(double(RAND_MAX)/2)-1;
            temp.push_back(val);
        }
        resul.push_back(temp);
    }
    return resul;
}


vector<vector<double> > creat(){   //create a M x N matrix by input method
    vector<vector<double> > resul;
    cout<<"input the row(s) and column(s) of the matrix which you want to create:";
    int row=0;
    int column=0;
    double val;
    cin>>row>>column;
    cout<<"please input your matrix, one by one:"<<endl;
    for(int crow=0;crow<row;crow++){
        vector<double> temp;
        for(int ccolumn=0;ccolumn<column;ccolumn++){
            cin>>val;
            temp.push_back(val);
        }
        resul.push_back(temp);
    }

    return resul;
}
vector<vector<double> > creat(double val, int row, int column){ //create a M x N matrix
    vector<vector<double> > resul;
    for(int crow=0;crow<row;crow++){
        vector<double> temp;
        for(int ccolumn=0;ccolumn<column;ccolumn++){
            temp.push_back(val);
        }
        resul.push_back(temp);
    }
    return resul;
}

vector<vector<double> > Hadamard(vector<vector<double> > a,vector<vector<double> > b){  //Hadamard function
    vector<vector<double> > error;
    vector<vector<double> > c;
    if(a.size()!=b.size()||a.size()==0||b.size()==0){  //row error check
       cout<<"error while product(row disagree): "<<a.size()<<"!="<<b.size()<<endl;
       return error;}
    for(int row=0;row<a.size();row++){
        if(a[row].size()!=b[row].size()){
            cout<<"error while product(column disagree):"<<a[row].size()<<"!="<<b[row].size()<<endl;
            return error;
        }
        vector<double> temp;
        for(int column=0;column<a[row].size();column++){   //process dot and column error check
            temp.push_back(a[row][column]*b[row][column]);
        }
        c.push_back(temp);
    }
    return c;
}

void display(vector<vector<double> > a){   // display matrix
    for(int row=0;row<a.size();row++){
        for(int column=0;column<a[row].size();column++){
            cout<<a[row][column]<<' ';
        }
        cout<<endl;
    }
    cout<<endl;
    //cout<<"display matrix complete"<<endl;
}

vector<vector<double> > dot(vector<vector<double> > a,vector<vector<double> > b){  //dot product
    vector<vector<double> > error;
    vector<vector<double> > c;
    if(a[0].size()!=b.size()){
        cout<<"error: can not dot product"<<endl;
        cout<<a.size()<<" "<<a[0].size()<<" "<<b.size()<<" "<<b[0].size()<<endl;
        return error;
    }
    for(int rowa=0;rowa<a.size();rowa++){        //calculate dot product using loops
        vector<double> temp;
        for(int columnb=0;columnb<b[0].size();columnb++){
            double resul=0;
            for(int num=0;num<b.size();num++){
                resul=a[rowa][num]*b[num][columnb]+resul;
            }
            temp.push_back(resul);
        }
        c.push_back(temp);
    }
    return c;
}

vector<vector<double> > transposition(vector<vector<double> > a){   //transposition matrix
    vector<vector<double> > c;
    for(int column=0;column<a[0].size();column++){
        vector<double> temp;
        for(int row=0;row<a.size();row++){
            temp.push_back(a[row][column]);
        }
        c.push_back(temp);
    }
    return c;
}


vector<vector<double> > reshape(vector<double> a, int column){ //reshape the matrix
    vector<vector<double> > resul;
    vector<double> temp;
    vector<double> null(0);
    //cout<<a.size();
    int j=0;
    for(int i=0; i<a.size(); i++){
        temp.push_back(a[i]);
        if(j==column-1){
            resul.push_back(temp);
            temp=null;
            j=0;
        }
        else
            j++;
    }
    return resul;
}

vector<vector<double> > vectorize(int a){
    if(a>9||a<0)
        cout<<"error"<<endl;
    //cout<<"vectorize test"<<endl;
    vector<vector<double> > resul(creat(0, 10, 1));
    //verify(resul);
    resul[a][0]=1;
    //verify(resul);
    //cout<<(resul[a-1][0])<<endl;
    return resul;
}

vector<vector<double> > vectorize_single(int a){
    vector<vector<double> > resul(creat(0, 1, 1));
    if(a==1)
        resul[0][0]=1;
    return resul;
}


int devectorize(vector<vector<double> > a){
    int resul=0;
    double k=a[0][0];
    for(int i=0; i<a.size(); i++){
        if(a[i][0]>k){
            k=a[i][0];
            resul=i;
        }
//    cout<<i<<":"<<resul<<endl;
    }
    return resul;
}

void verify(vector<vector<double> > a){
    cout<<"matrix size is:"<<a.size()<<" x ";
    if(a.size()>20)
        cout<<a[0].size()<<endl;
    else{
        for(int i=0; i<a.size(); i++)
            cout<<a[i].size()<<" ";
        cout<<endl;
        }
}

void verify_zero(vector<vector<double> > a){
    int num=0;
    for(int i=0; i<a.size(); i++){
        for(int j=0; j<a[i].size(); j++){
            if(a[i][j]!=0)
                num++;
        }
    }
    if(num==0)
        cout<<"all zero matrix!"<<endl;
    else
        cout<<num<<" non-zero element(s) existed"<<endl;
}

vector<int> shuffle(int n){   //create a vector with size n
    vector<int> data;
    for(int i=0; i<n; i++)
        data.push_back(i);
    srand(time(NULL));
    random_shuffle(data.begin(), data.end());
    return data;
}

vector<vector<double> > operator-(const vector<vector<double> > a, const vector<vector<double> > b){
    vector<vector<double> > resul;
    double num;
    if(a.size()!=b.size()||a.size()==0||b.size()==0){
        cout<<"can not add matrix(row disagree):"<<a.size()<<"!="<<b.size()<<endl;
        return resul;
    }
    if(a[0].size()!=b[0].size()){
        cout<<"can not add matrix(column disagree):"<<a[0].size()<<"!="<<b[0].size()<<endl;
        return resul;
    }
    for(int i=0;i<a.size();i++){
        vector<double> temp;
        for(int j=0;j<a[0].size();j++){
            num=a[i][j]-b[i][j];
            temp.push_back(num);
        }
        resul.push_back(temp);
    }
    return resul;
}

vector<vector<double> > operator+(const vector<vector<double> > a, const vector<vector<double> > b){
    vector<vector<double> > resul;
    double num;
    if(a.size()!=b.size()||a.size()==0||b.size()==0){
        cout<<"can not add matrix(row disagree):"<<a.size()<<"!="<<b.size()<<endl;
        return resul;
    }
    if(a[0].size()!=b[0].size()){
        cout<<"can not add matrix(column disagree):"<<a[0].size()<<"!="<<b[0].size()<<endl;
        return resul;
    }

    for(int i=0;i<a.size();i++){
        vector<double> temp;
        for(int j=0;j<a[0].size();j++){
            num=a[i][j]+b[i][j];
            temp.push_back(num);
        }
        resul.push_back(temp);
    }
    return resul;
}



vector<vector<double> > operator*(double a, const vector<vector<double> > b){
    vector<vector<double> > resul;
    double num;
    for(int i=0;i<b.size();i++){
        vector<double> temp;
        for(int j=0;j<b[0].size();j++){
            num=a*b[i][j];
            temp.push_back(num);
        }
        resul.push_back(temp);
    }
    return resul;
}
