using System;
using System.Diagnostics;

namespace FluidSimulationCoreCLR
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Starting Fluid Simulation...");

            // Initialize fluid simulation with 20x20x20
            FluidSimulation simulation = new FluidSimulation(new Vector3Int(20, 20, 20));

            // Run the simulation for a set number of frames
            const int frameCount = 100;

            Stopwatch stopwatch = new Stopwatch();

            for (int frame = 0; frame < frameCount; frame++)
            {
                stopwatch.Restart();

                // Update the simulation for one "frame" of computation
                simulation.Update();

                stopwatch.Stop();

                // Convert ticks to microseconds
                long microseconds = stopwatch.ElapsedTicks / (Stopwatch.Frequency / 1000000); 

                Console.WriteLine($"Frame {frame + 1}: {microseconds} µs");
            }

            Console.WriteLine("Fluid simulation complete.");
        }
    }
}
