using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;
using System.Diagnostics;
using System.Collections.Generic;
using System.IO;
using System;
using TMPro;

// This class runs the SPH calculations without Viusally rendering anything in Unity
// Additionally, it runs all calculations in sequence (rather than parallel) so that
// it is easier to compare to the CoreCLR equivalent code (also sequentially run)
public class SPH_Simulation_Pure_NO_Burst_No_Visuals : MonoBehaviour //Was previously called SPHSimulation
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

            // Run the calculations manually, without Burst and JobHandle
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

            // Swap particle buffers for the next iteration
            NativeArray<Particle> temp = particles;
            particles = newParticles;
            newParticles = temp;
        }
    }

    private void RunCalculations()
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

    private void ExportResults()
    {
        // Compute average times
        float first100Average = first100TotalTime / 100.0f;
        float remaining1000Average = remaining1000TotalTime / 1000.0f;

        // Log the average times to the Unity Console
        UnityEngine.Debug.Log($"SPH, {numberOfParticlesDesired} particles: Average time for the first 100 iterations: {first100Average} ms");
        UnityEngine.Debug.Log($"SPH, {numberOfParticlesDesired} particles: Average time for remaining 1000 iterations: {remaining1000Average} ms");

        // Update UI Texts
        if (averagesText != null)
        {
            averagesText.text = $"SPH, {numberOfParticlesDesired} particles:\n" +
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

    // Helper methods that perform calculations previously done in the job
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
        float secondDerivative = 90f / (math.PI * math.pow(radius, 5)) * (radius - distance);
        return viscosity * secondDerivative * (neighbor.velocity - particle.velocity) / neighbor.density;
    }

    private void ComputeDensityPressure(int index)
    {
        Particle particle = particles[index];
        float densitySum = 0f;

        int3 gridPosition = HashPosition(particle.position);
        for (int i = 0; i < neighborOffsets.Length; i++)
        {
            int3 neighborGridPosition = gridPosition + neighborOffsets[i];
            if (particleGridMap.TryGetFirstValue(neighborGridPosition, out int neighborIndex, out var iterator))
            {
                do
                {
                    Particle neighbor = particles[neighborIndex];
                    float3 diff = neighbor.position - particle.position;
                    float distanceSquared = math.lengthsq(diff);

                    if (distanceSquared < radius2)
                    {
                        densitySum += StandardKernel(distanceSquared);
                    }

                } while (particleGridMap.TryGetNextValue(out neighborIndex, ref iterator));
            }
        }

        particle.density = densitySum;
        particle.pressure = gasConstant * (particle.density - restingDensity);
        newParticles[index] = particle;
    }

    private void ComputeForces(int index)
    {
        Particle particle = particles[index];
        float3 pressureForce = float3.zero;
        float3 viscosityForce = float3.zero;

        int3 gridPosition = HashPosition(particle.position);
        for (int i = 0; i < neighborOffsets.Length; i++)
        {
            int3 neighborGridPosition = gridPosition + neighborOffsets[i];
            if (particleGridMap.TryGetFirstValue(neighborGridPosition, out int neighborIndex, out var iterator))
            {
                do
                {
                    Particle neighbor = particles[neighborIndex];
                    float3 diff = neighbor.position - particle.position;
                    float distanceSquared = math.lengthsq(diff);

                    if (distanceSquared < radius2 && neighborIndex != index)
                    {
                        float distance = math.sqrt(distanceSquared);
                        float3 direction = math.normalize(diff);
                        pressureForce += PressureKernelGradient(particle, neighbor, distance, direction);
                        viscosityForce += ViscosityKernel(particle, neighbor, distance);
                    }

                } while (particleGridMap.TryGetNextValue(out neighborIndex, ref iterator));
            }
        }

        float3 gravityForce = new float3(0, -9.8f, 0);
        particle.currentForce = pressureForce + viscosityForce + gravityForce;
        newParticles[index] = particle;
    }

    private void Integrate(int index)
    {
        Particle particle = particles[index];

        particle.velocity += timestep * particle.currentForce / particle.density;
        particle.position += timestep * particle.velocity;

        if (particle.position.x - radius < 0)
        {
            particle.velocity.x *= boundDamping;
            particle.position.x = radius;
        }
        if (particle.position.x + radius > boxSize.x)
        {
            particle.velocity.x *= boundDamping;
            particle.position.x = boxSize.x - radius;
        }
        if (particle.position.y - radius < 0)
        {
            particle.velocity.y *= boundDamping;
            particle.position.y = radius;
        }
        if (particle.position.y + radius > boxSize.y)
        {
            particle.velocity.y *= boundDamping;
            particle.position.y = boxSize.y - radius;
        }
        if (particle.position.z - radius < 0)
        {
            particle.velocity.z *= boundDamping;
            particle.position.z = radius;
        }
        if (particle.position.z + radius > boxSize.z)
        {
            particle.velocity.z *= boundDamping;
            particle.position.z = boxSize.z - radius;
        }

        newParticles[index] = particle;
    }
}


// SimulationResults class to store the results
[Serializable]
public class SimulationResults
{
    public float First100Average;
    public float Remaining1000Average;
    public long[] IndividualIterationTimes;
    public int ParticleCount;
}
