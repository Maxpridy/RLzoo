#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>
#include <iostream>
#include <numeric>
#include <execution>


int work(int id) {
    id += 1;
    return id;
}

int main() {
    const auto start_time = std::chrono::steady_clock::now();
    
    int sum_value = 0;

    std::vector<std::future<int>> futures;
    for (int j = 0; j < 1000; j++) {
        for (int i = 0; i < 1000; i++) {
            futures.emplace_back(std::async(std::launch::async, work, i));
        }
    }

    for (auto& f : futures) {
        sum_value += f.get();
    }

    const std::chrono::duration<double> dur = std::chrono::steady_clock::now() - start_time;

    std::cout << dur.count() << std::endl;
    std::cout << sum_value << std::endl;
}