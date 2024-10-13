using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using System.Diagnostics;
using System.Collections.Generic;
using System.IO;
using System;
using TMPro;

// Ensure the script is attached to a GameObject in your Unity scene
public class SPHSimulation : MonoBehaviour
{
    private NativeArray<Particle> particles;
    private NativeArray<Particle> newParticles;
    private NativeParallelMultiHashMap<int3, int> particleGridMap;
    private NativeArray<int3> neighborOffsets;

    // Fluid constants
    private const float particleMass = 1f;
    private const float gasConstant = 2f;
    private const float restingDensity = 1f;
    private const float viscosity = -0.003f;
    private const float timestep = 0.007f;
    private const float boundDamping = -0.3f;
    private float3 boxSize = new float3(4, 10, 3);
    private const float radius = 0.1f;
    private float radius2 => radius * radius;

    private Stopwatch stopwatch;
    private long first100TotalTime = 0;
    private long remaining1000TotalTime = 0;
    private List<long> iterationTimes; // To store individual iteration times

    public TextMeshProUGUI resultsCompletedText;

    private void Start()
    {
        int numberOfParticles = 500;
        particles = new NativeArray<Particle>(numberOfParticles, Allocator.Persistent);
        newParticles = new NativeArray<Particle>(numberOfParticles, Allocator.Persistent);

        // Initialize neighbor offsets
        neighborOffsets = new NativeArray<int3>(27, Allocator.Persistent);
        InitializeNeighborOffsets();

        // Initialize particle grid map
        particleGridMap = new NativeParallelMultiHashMap<int3, int>(numberOfParticles, Allocator.Persistent);

        // Initialize the list to store iteration times
        iterationTimes = new List<long>(1100);

        // Spawn particles
        SpawnParticlesInBox(numberOfParticles);

        // Initialize stopwatch for performance measurement
        stopwatch = new Stopwatch();

        // Run the simulation with timing
        for (int i = 0; i < 1100; i++)
        {
            stopwatch.Restart();

            RunCalculations();

            stopwatch.Stop();

            long elapsedTime = stopwatch.ElapsedMilliseconds;
            iterationTimes.Add(elapsedTime);

            // Record timings for first 100 and remaining 1000 iterations
            if (i < 100)
            {
                first100TotalTime += elapsedTime;
            }
            else
            {
                remaining1000TotalTime += elapsedTime;
            }
        }

        // Compute average times
        float first100Average = first100TotalTime / 100.0f;
        float remaining1000Average = remaining1000TotalTime / 1000.0f;

        // Print the average times to the Unity Console
        UnityEngine.Debug.Log($"Burst SPH: Average time for the first 100 iterations: {first100Average} ms");
        UnityEngine.Debug.Log($"Burst SPH: Average time for remaining 1000 iterations: {remaining1000Average} ms");

        // Create a data object to hold the results
        SimulationResults results = new SimulationResults
        {
            First100Average = first100Average,
            Remaining1000Average = remaining1000Average,
            IndividualIterationTimes = iterationTimes.ToArray()
        };

        // Serialize the results to JSON
        string json = JsonUtility.ToJson(results, true);

        // Define the file path (writes to the project's persistent data path)
        string filePath = Path.Combine(Application.persistentDataPath, "sph_simulation_results.json");

        // Write the JSON to the file
        try
        {
            File.WriteAllText(filePath, json);
            UnityEngine.Debug.Log($"Simulation results successfully written to: {filePath}");
            resultsCompletedText.text = $"Simulation results successfully written to: {filePath}";

        }
        catch (Exception ex)
        {
            UnityEngine.Debug.LogError($"Failed to write simulation results to JSON. Exception: {ex.Message}");
        }

        UnityEngine.Debug.Log("Simulation complete!");
    }

    private void OnDestroy()
    {
        if (particles.IsCreated)
            particles.Dispose();
        if (newParticles.IsCreated)
            newParticles.Dispose();
        if (neighborOffsets.IsCreated)
            neighborOffsets.Dispose();
        if (particleGridMap.IsCreated)
            particleGridMap.Dispose();
    }

    private void SpawnParticlesInBox(int numParticles)
    {
        Unity.Mathematics.Random random = new Unity.Mathematics.Random(1234);
        for (int i = 0; i < numParticles; i++)
        {
            float3 position = new float3(
                random.NextFloat(0, boxSize.x),
                random.NextFloat(0, boxSize.y),
                random.NextFloat(0, boxSize.z)
            );

            particles[i] = new Particle
            {
                position = position,
                velocity = float3.zero,
                currentForce = float3.zero,
                density = restingDensity,
                pressure = 0
            };
        }
    }

    private void InitializeNeighborOffsets()
    {
        int index = 0;
        for (int x = -1; x <= 1; x++)
        {
            for (int y = -1; y <= 1; y++)
            {
                for (int z = -1; z <= 1; z++)
                {
                    neighborOffsets[index] = new int3(x, y, z);
                    index++;
                }
            }
        }
    }

    [BurstCompile]
    private void RunCalculations()
    {
        particleGridMap.Clear();

        // Populate the particle grid map with current particle positions
        for (int i = 0; i < particles.Length; i++)
        {
            int3 gridPos = HashPosition(particles[i].position);
            particleGridMap.Add(gridPos, i);
        }

        // Execute density and pressure computation
        for (int i = 0; i < particles.Length; i++)
        {
            ComputeDensityPressure(i);
        }

        // Execute force computation
        for (int i = 0; i < particles.Length; i++)
        {
            ComputeForces(i);
        }

        // Execute integration step
        for (int i = 0; i < particles.Length; i++)
        {
            Integrate(i);
        }

        // Swap particle buffers for the next frame
        NativeArray<Particle> temp = particles;
        particles = newParticles;
        newParticles = temp;
    }

    private int3 HashPosition(float3 position)
    {
        int x = (int)math.floor(position.x / (radius * 2.5f));
        int y = (int)math.floor(position.y / (radius * 2.5f));
        int z = (int)math.floor(position.z / (radius * 2.5f));
        return new int3(x, y, z);
    }

    private void ComputeDensityPressure(int id)
    {
        Particle particle = particles[id];
        float3 origin = particle.position;
        float densitySum = 0;

        int3 gridPos = HashPosition(origin);

        for (int i = 0; i < neighborOffsets.Length; i++)
        {
            int3 neighborPos = gridPos + neighborOffsets[i];
            if (particleGridMap.TryGetFirstValue(neighborPos, out int neighborIndex, out var iterator))
            {
                do
                {
                    Particle neighbor = particles[neighborIndex];
                    float3 diff = origin - neighbor.position;
                    float distanceSquared = math.lengthsq(diff);

                    if (distanceSquared < radius2)
                    {
                        densitySum += StandardKernel(distanceSquared);
                    }
                }
                while (particleGridMap.TryGetNextValue(out neighborIndex, ref iterator));
            }
        }

        particle.density = densitySum * particleMass;
        particle.pressure = gasConstant * (particle.density - restingDensity);
        newParticles[id] = particle;
    }

    private float StandardKernel(float distanceSquared)
    {
        float x = 1.0f - distanceSquared / radius2;
        return 315f / (64f * math.PI * math.pow(radius2, 1.5f)) * x * x * x;
    }

    private void ComputeForces(int id)
    {
        Particle particle = particles[id];
        float3 origin = particle.position;
        float3 pressureForce = float3.zero;
        float3 viscousForce = float3.zero;

        int3 gridPos = HashPosition(origin);

        for (int i = 0; i < neighborOffsets.Length; i++)
        {
            int3 neighborPos = gridPos + neighborOffsets[i];
            if (particleGridMap.TryGetFirstValue(neighborPos, out int neighborIndex, out var iterator))
            {
                do
                {
                    if (neighborIndex == id) continue;

                    Particle neighbor = particles[neighborIndex];
                    float3 diff = origin - neighbor.position;
                    float distanceSquared = math.lengthsq(diff);

                    if (distanceSquared < radius2)
                    {
                        float distance = math.sqrt(distanceSquared);
                        pressureForce += PressureKernelGradient(particle, neighbor, distance, diff);
                        viscousForce += ViscosityKernel(particle, neighbor, distance);
                    }
                }
                while (particleGridMap.TryGetNextValue(out neighborIndex, ref iterator));
            }
        }

        // External force (gravity)
        float3 externalForce = new float3(0, -9.81f * particleMass, 0);
        particle.currentForce = externalForce - pressureForce + viscousForce;
        newParticles[id] = particle;
    }

    private float3 PressureKernelGradient(Particle particle, Particle neighbor, float distance, float3 direction)
    {
        float gradValue = -45.0f / (math.PI * math.pow(radius, 4)) * math.pow(1.0f - distance / radius, 2);
        return gradValue * direction * (particle.pressure + neighbor.pressure) / (2 * neighbor.density);
    }

    private float3 ViscosityKernel(Particle particle, Particle neighbor, float distance)
    {
        float secondDerivative = 90f / (math.PI * math.pow(radius, 5)) * (1.0f - distance / radius);
        return viscosity * secondDerivative * (neighbor.velocity - particle.velocity) / neighbor.density;
    }

    private void Integrate(int id)
    {
        Particle particle = particles[id];
        particle.velocity += (particle.currentForce / particleMass) * timestep;
        particle.position += particle.velocity * timestep;

        // Handle bounding box collisions
        float3 min = -boxSize / 2;
        float3 max = boxSize / 2;

        // Get the current velocity of the particle
        float3 velocity = particle.velocity;

        // Check and modify the X component of the velocity
        if (particle.position.x < min.x || particle.position.x > max.x)
        {
            velocity.x *= boundDamping;
        }

        // Check and modify the Y component of the velocity
        if (particle.position.y < min.y || particle.position.y > max.y)
        {
            velocity.y *= boundDamping;
        }

        // Check and modify the Z component of the velocity
        if (particle.position.z < min.z || particle.position.z > max.z)
        {
            velocity.z *= boundDamping;
        }

        // Update the particle's velocity
        particle.velocity = velocity;

        newParticles[id] = particle;
    }

    // Struct representing a particle
    private struct Particle
    {
        public float3 position;
        public float3 velocity;
        public float3 currentForce;
        public float density;
        public float pressure;
    }

    // Serializable class for JSON export
    [Serializable]
    private class SimulationResults
    {
        public float First100Average;
        public float Remaining1000Average;
        public long[] IndividualIterationTimes;
    }

    // Method to write results to JSON (included within Start for simplicity)
    // Alternatively, you could separate this logic if desired
}
