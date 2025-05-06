#include<bits/stdc++.h>
#include<chrono>
#include<omp.h>

using namespace std;
using namespace std::chrono;

vector<int> generate_random_vec(int size){
    vector<int> vec(size);
    srand(time(0));
    for(int i = 0; i < size; i++){
        vec[i] = rand() % 10000;
    }
    return vec;
}

void sequential_bubblesort(vector<int>& arr){
    int n = arr.size();
    for(int i = 0; i < n-1; i++){
        for(int j = 0; j < n-1-i; j++){
            if(arr[j] > arr[j+1])swap(arr[j], arr[j + 1]);
        }
    }
}

void parallel_bubblesort(vector<int>& arr){
    int n = arr.size();
    for(int i = 0; i < n; i++){

        #pragma omp parallel for shared(arr, i)
        for(int j = i % 2; j < n-1; j+=2){
            if(arr[j] > arr[j+1])swap(arr[j], arr[j+1]);
        }
    }    
}

void merge(vector<int>& arr, int left, int mid, int right){
    vector<int> leftarr(arr.begin() + left, arr.begin() + mid + 1);
    vector<int> rightarr(arr.begin() + mid + 1, arr.begin() + right + 1);

    int i = 0, j = 0, k = left;
    // Index for leftarr, rightarr & arr.

    while(i < leftarr.size() && j < rightarr.size()){
        if(leftarr[i] <=  rightarr[j])arr[k++] = leftarr[i++];
        else arr[k++] = rightarr[j++];
    }
    while(i < leftarr.size())arr[k++] = leftarr[i++];
    while(j < rightarr.size())arr[k++] = rightarr[j++];
}

void sequential_mergesort(vector<int>& arr, int left, int right){
    if(left < right){
        int mid = (left+right)/2;
        sequential_mergesort(arr, left, mid);
        sequential_mergesort(arr, mid+1, right);
        merge(arr, left, mid, right);
    }
}

void parallel_mergesort(vector<int>& arr, int left, int right, int depth = 0){
    if(left < right){
        int mid = (left+right)/2;

        if(depth < 4){
        #pragma omp parallel sections
        {
            #pragma omp section
            parallel_mergesort(arr, left, mid, depth + 1);
            #pragma omp section
            parallel_mergesort(arr, mid+1, right, depth + 1);
        }
        }
        else{
            sequential_mergesort(arr, left, mid);
            sequential_mergesort(arr, mid+1, right);
        }
        merge(arr, left, mid, right);
    }
}



int main(){
    cout<<"Enter size of vector: ";
    int n; cin>>n;
    
    vector<int> vec = generate_random_vec(n);
    vector<int> arr = vec;

    auto start = high_resolution_clock::now();
    sequential_bubblesort(arr);
    auto end = high_resolution_clock::now();
    auto total_duration = duration_cast<milliseconds>(end - start);
    cout<<"Time take for sequential bubble sort: "<<total_duration.count()<<"ms\n";

    arr = vec;
    start = high_resolution_clock::now();
    parallel_bubblesort(arr);
    end = high_resolution_clock::now();
    total_duration = duration_cast<milliseconds>(end - start);
    cout<<"Time take for parallel bubble sort: "<<total_duration.count()<<"ms\n";

    arr = vec;
    auto start_time = high_resolution_clock::now();
    sequential_mergesort(arr, 0, n-1);
    auto end_time = high_resolution_clock::now();
    auto total_duration_merge = duration_cast<milliseconds>(end_time - start_time);
    cout<<"Time take for sequential merge sort: "<<total_duration_merge.count()<<"ms\n";    

    arr = vec;
    start_time = high_resolution_clock::now();   
    parallel_mergesort(arr, 0, n-1); 
    end_time = high_resolution_clock::now();
    total_duration_merge = duration_cast<milliseconds>(end_time - start_time);
    cout<<"Time take for parallel merge sort: "<<total_duration_merge.count()<<"ms\n";    

}