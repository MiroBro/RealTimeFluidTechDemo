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

    // UI Elements to display results
    public TextMeshProUGUI resultsCompletedText;
    public TextMeshProUGUI averagesText;

    [SerializeField]
    public int numberOfParticlesDesired = 8000;

    private void Start()
    {
        InitializeSimulation();
        RunSimulation();
        ExportResults();
    }

    private void OnDestroy()
    {
        DisposeNativeCollections();
    }

    private void InitializeSimulation()
    {
        int numberOfParticles = numberOfParticlesDesired;
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
    }

    private void RunSimulation()
    {
        for (int i = 0; i < 1100; i++)
        {
            stopwatch.Restart();

            // Create and schedule the RunCalculationsJob
            RunCalculationsJob runCalculationsJob = new RunCalculationsJob
            {
                particles = particles,
                newParticles = newParticles,
                particleGridMap = particleGridMap,
                neighborOffsets = neighborOffsets,
                radius2 = radius2,
                particleMass = particleMass,
                gasConstant = gasConstant,
                restingDensity = restingDensity,
                timestep = timestep,
                boundDamping = boundDamping,
                boxSize = boxSize,
                viscosity = viscosity
            };

            JobHandle jobHandle = runCalculationsJob.Schedule();
            jobHandle.Complete();

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

            // Swap particle buffers for the next iteration
            NativeArray<Particle> temp = particles;
            particles = newParticles;
            newParticles = temp;
        }
    }

    private void ExportResults()
    {
        // Compute average times
        float first100Average = first100TotalTime / 100.0f;
        float remaining1000Average = remaining1000TotalTime / 1000.0f;

        // Log the average times to the Unity Console
        UnityEngine.Debug.Log($"Burst SPH, {numberOfParticlesDesired} particles: Average time for the first 100 iterations: {first100Average} ms");
        UnityEngine.Debug.Log($"Burst SPH, {numberOfParticlesDesired} particles: Average time for remaining 1000 iterations: {remaining1000Average} ms");

        // Update UI Texts
        if (averagesText != null)
        {
            averagesText.text = $"Burst SPH, {numberOfParticlesDesired} particles:\n" +
                                $"Average time for first 100 iterations: {first100Average} ms\n" +
                                $"Average time for remaining 1000 iterations: {remaining1000Average} ms";
        }

        // Create a data object to hold the results
        SimulationResults results = new SimulationResults
        {
            First100Average = first100Average,
            Remaining1000Average = remaining1000Average,
            IndividualIterationTimes = iterationTimes.ToArray(),
            ParticleCount = numberOfParticlesDesired  // Set the particle count
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

            if (resultsCompletedText != null)
            {
                resultsCompletedText.text = $"Simulation results successfully written to:\n{filePath}";
            }
        }
        catch (Exception ex)
        {
            UnityEngine.Debug.LogError($"Failed to write simulation results to JSON. Exception: {ex.Message}");
            if (resultsCompletedText != null)
            {
                resultsCompletedText.text = $"Failed to write simulation results to JSON.\nException: {ex.Message}";
            }
        }

        UnityEngine.Debug.Log("Simulation complete!");
    }


    private void DisposeNativeCollections()
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

    // Job Struct to perform all calculations sequentially with Burst compilation
    [BurstCompile]
    private struct RunCalculationsJob : IJob
    {
        [ReadOnly]
        public NativeArray<Particle> particles;

        public NativeArray<Particle> newParticles;

        public NativeParallelMultiHashMap<int3, int> particleGridMap;

        [ReadOnly]
        public NativeArray<int3> neighborOffsets;

        [ReadOnly]
        public float radius2;

        [ReadOnly]
        public float particleMass;

        [ReadOnly]
        public float gasConstant;

        [ReadOnly]
        public float restingDensity;

        [ReadOnly]
        public float timestep;

        [ReadOnly]
        public float boundDamping;

        [ReadOnly]
        public float3 boxSize;

        [ReadOnly]
        public float viscosity;

        public void Execute()
        {
            // Clear the particle grid map
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
        }

        private static int3 HashPosition(float3 position)
        {
            float inv = 1.0f / (radius * 2.5f);
            int x = (int)math.floor(position.x * inv);
            int y = (int)math.floor(position.y * inv);
            int z = (int)math.floor(position.z * inv);
            return new int3(x, y, z);
        }

        private float StandardKernel(float distanceSquared)
        {
            float x = 1.0f - distanceSquared / radius2;
            return 315f / (64f * math.PI * math.pow(radius2, 1.5f)) * x * x * x;
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

        private void ComputeDensityPressure(int id)
        {
            Particle particle = particles[id];
            float3 origin = particle.position;
            float densitySum = 0f;

            int3 gridPos = HashPosition(origin);

            for (int i = 0; i < neighborOffsets.Length; i++)
            {
                int3 neighborPos = gridPos + neighborOffsets[i];
                NativeParallelMultiHashMapIterator<int3> iterator;
                if (particleGridMap.TryGetFirstValue(neighborPos, out int neighborIndex, out iterator))
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
                NativeParallelMultiHashMapIterator<int3> iterator;
                if (particleGridMap.TryGetFirstValue(neighborPos, out int neighborIndex, out iterator))
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
                            if (distance > 0f)
                            {
                                float3 pressureGrad = PressureKernelGradient(particle, neighbor, distance, diff);
                                pressureForce += pressureGrad;

                                float3 viscousGrad = ViscosityKernel(particle, neighbor, distance);
                                viscousForce += viscousGrad;
                            }
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
    }

    // Struct representing a particle
    [Serializable]
    public struct Particle
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
        public int ParticleCount;  // New field to store the number of particles
        public float First100Average;
        public float Remaining1000Average;
        public long[] IndividualIterationTimes;
    }

}
