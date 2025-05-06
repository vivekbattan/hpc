#include<bits/stdc++.h>
#include<chrono>
#include<omp.h>

using namespace std;
using namespace std::chrono;

int main(){
    int n;
    cout<<"Enter size of array: ";
    cin>>n;
    //Vector with random variables of size n;
    vector<int> arr(n);
    srand(time(0));
    for(int i = 0 ; i < n; i++){
        arr[i] = rand() % 1000 + 1;
    }

    //Minimum
    int min_val = INT_MAX;
    auto start = high_resolution_clock::now();

    #pragma omp parallel for reduction(min:min_val)
    for(int i = 0; i < n; i++){
        if(arr[i] < min_val)min_val = arr[i];
    }

    auto end = high_resolution_clock::now();
    cout<<"Minimum: "<<min_val<<"\nTime taken for finding minimum using parallel reduction: "<<duration_cast<microseconds>(end-start).count()<<" us\n";

    //Maximum
    int max_val = INT_MIN;
    start = high_resolution_clock::now();

    #pragma omp parallel for reduction(max:max_val)
    for(int i = 0; i < n; i++){
        if(arr[i] > max_val)max_val = arr[i];
    }

    end = high_resolution_clock::now();
    cout<<"Maximum: "<<max_val<<"\nTime taken for finding maximum using parallel reduction: "<<duration_cast<microseconds>(end-start).count()<<" us\n";

    //Addition
    long long sum = 0;
    start = high_resolution_clock::now();

    #pragma omp parallel for reduction(+:sum)
    for(int i = 0; i < n; i++){
        sum += arr[i];
    }

    end = high_resolution_clock::now();
    cout<<"Sum: "<<sum<<"\nTime taken for finding sum using parallel reduction: "<<duration_cast<microseconds>(end-start).count()<<" us\n";

    //Average
    auto start_time = high_resolution_clock::now();
    double avg = 0.0;
    avg = static_cast<double>(sum) / n;
    auto end_time = high_resolution_clock::now();
    cout<<"Average: "<<avg<<"\nTime taken for finding average using parallel reduction: "<<duration_cast<microseconds>(end-start+end_time-start_time).count()<<" us\n";
}