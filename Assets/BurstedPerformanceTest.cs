using System;
using Unity.Burst;
using UnityEngine;
using Unity.Collections;
using Unity.Jobs;
using System.Diagnostics;

public class BurstedPerformanceTest : MonoBehaviour
{
    private Stopwatch stopwatch;

    private void Start()
    {
        stopwatch = new Stopwatch();

        // Measure different function sizes
        MeasureBurstedPerformance(100);
        MeasureBurstedPerformance(1000);
        MeasureBurstedPerformance(10000);
        MeasureBurstedPerformance(100000);
    }

    private void MeasureBurstedPerformance(int iterations)
    {
        UnityEngine.Debug.Log($"Testing Bursted code with {iterations} iterations");

        stopwatch.Reset();
        stopwatch.Start();

        var job = new BurstedJob() { iterations = iterations };
        job.Schedule().Complete();

        stopwatch.Stop();
        UnityEngine.Debug.Log($"Bursted code took: {stopwatch.ElapsedMilliseconds} ms for {iterations} iterations");
    }

    [BurstCompile]
    private struct BurstedJob : IJob
    {
        public int iterations;

        public void Execute()
        {
            for (int i = 0; i < iterations; i++)
            {
                SmallFunction(); // This can be replaced with MediumFunction, LargeFunction, etc.
            }
        }

        [BurstCompile] // Small function
        private void SmallFunction()
        {
            // A trivial function that doesn't do much
            int x = 10;
            int y = x * 2;
        }

        [BurstCompile] // Medium function
        private void MediumFunction()
        {
            // A function with moderate complexity
            for (int i = 0; i < 100; i++)
            {
                int x = i * 2;
                int y = x * 3;
            }
        }

        [BurstCompile] // Large function
        private void LargeFunction()
        {
            // A function with a lot of computation
            int result = 0;
            for (int i = 0; i < 10000; i++)
            {
                result += i * 2;
            }
        }
    }
}
