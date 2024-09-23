using System.Collections;
using System.Collections.Generic;
using System.Security.Cryptography;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;

public class SPH_MiroCsharp_Bursted_Optimized : MonoBehaviour
{
    [Header("General")]
    public Transform collisionSphere;
    public bool showSpheres = true;
    public Vector3Int numToSpawn = new Vector3Int(10, 10, 10);
    private int totalParticles => numToSpawn.x * numToSpawn.y * numToSpawn.z;

    public Vector3 boxSize = new Vector3(4, 10, 3);
    public Vector3 spawnCenter;
    public float particleRadius = 0.1f;
    public float spawnJitter = 0.2f;

    [Header("Particle Rendering")]
    public Mesh particleMesh;
    public float particleRenderSize = 8f;
    public Material material;

    [Header("Fluid Constants")]
    public float boundDamping = -0.3f;
    public float viscosity = -0.003f;
    public float particleMass = 1f;
    public float gasConstant = 2f;
    public float restingDensity = 1f;
    public float timestep = 0.007f;
    public float radius => particleRadius;
    public float radius2 => particleRadius * particleRadius;

    // NativeArray to store particles for use in jobs (Optimized)
    private NativeArray<Particle> particles;
    private NativeArray<Particle> newParticles;

    // Particle struct
    private struct Particle
    {
        public float pressure;
        public float density;
        public Vector3 currentForce;
        public Vector3 velocity;
        public Vector3 position;
    }

    private List<Matrix4x4> matrices;

    private void Start()
    {
        if (material != null)
        {
            material.enableInstancing = true;
        }

    }

    private void Update()
    {
        if (showSpheres)
        {
            RenderParticles();
        }
    }

    private void FixedUpdate()
    {
        // Create and populate the spatial hash grid
        NativeParallelMultiHashMap<int, int> grid = new NativeParallelMultiHashMap<int, int>(totalParticles, Allocator.TempJob);
        float cellSize = radius * 2; // Adjust the cell size based on particle interaction radius

        // Populate the grid
        for (int i = 0; i < totalParticles; i++)
        {
            int3 cellIndex = GetCellIndex(particles[i].position, cellSize);
            int hash = HashCellIndex(cellIndex);
            grid.Add(hash, i);
        }


        // Scheduling jobs to run in parallel
        var densityJob = new ComputeDensityPressureJob
        {
            particles = particles,
            newParticles = newParticles,
            particleMass = particleMass,
            gasConstant = gasConstant,
            restingDensity = restingDensity,
            radius2 = radius2,
            grid = grid, // Pass the spatial hash grid
            cellSize = cellSize // Pass the cell size
        };
        JobHandle densityHandle = densityJob.Schedule(totalParticles, 64);

        var forceJob = new ComputeForcesJob
        {
            particles = newParticles,
            newParticles = particles,
            particleMass = particleMass,
            viscosity = viscosity,
            radius = radius,
            collisionSphereCenter = collisionSphere.position,
            collisionSphereRadius = collisionSphere.localScale.x / 2,
            grid = grid, // Pass the spatial hash grid
            cellSize = cellSize // Pass the cell size
        };
        JobHandle forceHandle = forceJob.Schedule(totalParticles, 64, densityHandle);

        var integrationJob = new IntegrateJob
        {
            particles = particles,
            newParticles = newParticles,
            timestep = timestep,
            particleMass = particleMass,
            boundDamping = boundDamping,
            boxSize = boxSize,
            radius = radius
        };
        JobHandle integrationHandle = integrationJob.Schedule(totalParticles, 64, forceHandle);

        integrationHandle.Complete();
        // Dispose of the grid after all jobs are complete
        grid.Dispose();


        // Swap the buffers for the next frame
        var temp = particles;
        particles = newParticles;
        newParticles = temp;
    }

    private void Awake()
    {
        particles = new NativeArray<Particle>(totalParticles, Allocator.Persistent);
        newParticles = new NativeArray<Particle>(totalParticles, Allocator.Persistent);
        SpawnParticlesInBox();
        InitializeMatrices();
    }

    private void OnDestroy()
    {
        if (particles.IsCreated)
        {
            particles.Dispose(); // Dispose of NativeArray (Memory management)
        }
        if (newParticles.IsCreated)
        {
            newParticles.Dispose(); // Dispose of NativeArray (Memory management)
        }
    }

    public static int3 GetCellIndex(Vector3 position, float cellSize)
    {
        return new int3(Mathf.FloorToInt(position.x / cellSize),
                        Mathf.FloorToInt(position.y / cellSize),
                        Mathf.FloorToInt(position.z / cellSize));
    }

    public static int HashCellIndex(int3 cellIndex)
    {
        // A simple hash function for 3D grid cells
        return (cellIndex.x * 73856093) ^ (cellIndex.y * 19349663) ^ (cellIndex.z * 83492791);
    }


    private void InitializeParticles()
    {
        for (int i = 0; i < totalParticles; i++)
        {
            particles[i] = new Particle
            {
                position = spawnCenter + new Vector3(
                    (i % numToSpawn.x) * particleRadius * 2,
                    ((i / numToSpawn.x) % numToSpawn.y) * particleRadius * 2,
                    (i / (numToSpawn.x * numToSpawn.x)) * particleRadius * 2),
                velocity = Vector3.zero,
                currentForce = Vector3.zero,
                density = restingDensity,
                pressure = 0
            };
        }
    }

    private void InitializeMatrices()
    {
        matrices = new List<Matrix4x4>(totalParticles);
        for (int i = 0; i < totalParticles; i++)
        {
            matrices.Add(Matrix4x4.identity);
        }
    }

    private void RenderParticles()
    {
        for (int i = 0; i < totalParticles; i++)
        {
            matrices[i] = Matrix4x4.TRS(particles[i].position, Quaternion.identity, Vector3.one * particleRenderSize);
        }
        Graphics.DrawMeshInstanced(particleMesh, 0, material, matrices);
    }

    private void SpawnParticlesInBox()
    {
        Vector3 spawnPoint = spawnCenter;

        int index = 0;
        for (int x = 0; x < numToSpawn.x; x++)
        {
            for (int y = 0; y < numToSpawn.y; y++)
            {
                for (int z = 0; z < numToSpawn.z; z++)
                {
                    Vector3 spawnPos = spawnPoint + new Vector3(
                        x * particleRadius * 2,
                        y * particleRadius * 2,
                        z * particleRadius * 2);

                    spawnPos += UnityEngine.Random.insideUnitSphere * particleRadius * spawnJitter;

                    particles[index] = new Particle
                    {
                        position = spawnPos,
                        velocity = Vector3.zero,
                        currentForce = Vector3.zero,
                        density = restingDensity,
                        pressure = 0
                    };

                    index++;
                }
            }
        }
    }

    [BurstCompile]
    private struct ComputeDensityPressureJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<Particle> particles;
        public NativeArray<Particle> newParticles;
        public float particleMass;
        public float gasConstant;
        public float restingDensity;
        public float radius2;

        [ReadOnly] public NativeParallelMultiHashMap<int, int> grid; // Mark as ReadOnly
        public float cellSize;

        private float StdKernel(float distanceSquared)
        {
            float x = 1.0f - distanceSquared / radius2;
            return 315f / (64f * Mathf.PI * Mathf.Pow(radius2, 1.5f)) * x * x * x;
        }

        public void Execute(int id)
        {
            Particle particle = particles[id];
            Vector3 origin = particle.position;
            float sum = 0;

            int3 originCell = GetCellIndex(origin, cellSize);

            // Loop through surrounding cells (-1, 0, 1) range in each axis
            for (int x = -1; x <= 1; x++)
            {
                for (int y = -1; y <= 1; y++)
                {
                    for (int z = -1; z <= 1; z++)
                    {
                        int3 neighborCell = originCell + new int3(x, y, z);
                        int hash = HashCellIndex(neighborCell);

                        // Iterate over all particles in this grid cell
                        if (grid.TryGetFirstValue(hash, out int otherId, out NativeParallelMultiHashMapIterator<int> it))
                        {
                            do
                            {
                                Vector3 diff = origin - particles[otherId].position;
                                float distanceSquared = Vector3.Dot(diff, diff);

                                if (radius2 > distanceSquared)
                                {
                                    sum += StdKernel(distanceSquared);
                                }

                            } while (grid.TryGetNextValue(out otherId, ref it));
                        }
                    }
                }
            }

            particle.density = sum * particleMass;
            particle.pressure = gasConstant * (particle.density - restingDensity);
            newParticles[id] = particle;
        }
    }

    [BurstCompile]
    private struct ComputeForcesJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<Particle> particles;
        public NativeArray<Particle> newParticles;
        public float particleMass;
        public float viscosity;
        public float radius;
        public Vector3 collisionSphereCenter;
        public float collisionSphereRadius;

        [ReadOnly] public NativeParallelMultiHashMap<int, int> grid; // Mark as ReadOnly
        public float cellSize;

        private float SpikyKernelFirstDerivative(float distance)
        {
            float x = 1.0f - distance / radius;
            return -45.0f / (Mathf.PI * Mathf.Pow(radius, 4)) * x * x;
        }

        private float SpikyKernelSecondDerivative(float distance)
        {
            float x = 1.0f - distance / radius;
            return 90f / (Mathf.PI * Mathf.Pow(radius, 5)) * x;
        }

        private Vector3 SpikyKernelGradient(float distance, Vector3 directionFromCenter)
        {
            return SpikyKernelFirstDerivative(distance) * directionFromCenter;
        }

        public void Execute(int id)
        {
            Particle particle = particles[id];
            Vector3 origin = particle.position;
            float density2 = particle.density * particle.density;
            Vector3 pressure = Vector3.zero;
            Vector3 visc = Vector3.zero;
            float mass2 = particleMass * particleMass;

            int3 originCell = GetCellIndex(origin, cellSize);

            // Loop through surrounding cells (-1, 0, 1) range in each axis
            for (int x = -1; x <= 1; x++)
            {
                for (int y = -1; y <= 1; y++)
                {
                    for (int z = -1; z <= 1; z++)
                    {
                        int3 neighborCell = originCell + new int3(x, y, z);
                        int hash = HashCellIndex(neighborCell);

                        // Iterate over all particles in this grid cell
                        if (grid.TryGetFirstValue(hash, out int otherId, out NativeParallelMultiHashMapIterator<int> it))
                        {
                            do
                            {
                                if (otherId == id) continue;

                                Vector3 diff = origin - particles[otherId].position;
                                float distSquared = diff.sqrMagnitude;

                                if (distSquared < radius * radius)
                                {
                                    float distance = Mathf.Sqrt(distSquared);
                                    Vector3 direction = diff.normalized;

                                    Vector3 pressureGradientDirection = Vector3.Normalize(particle.position - particles[otherId].position);
                                    pressure += mass2 * (particle.pressure / density2 + particles[otherId].pressure / (particles[otherId].density * particles[otherId].density)) * SpikyKernelGradient(distSquared, pressureGradientDirection);
                                    visc += viscosity * mass2 * (particles[otherId].velocity - particle.velocity) / particles[otherId].density * SpikyKernelSecondDerivative(distance);
                                }

                            } while (grid.TryGetNextValue(out otherId, ref it));
                        }
                    }
                }
            }

            particle.currentForce = new Vector3(0, (-9.81f * particleMass), 0) - pressure + visc;

            Vector3 colDir = particle.position - collisionSphereCenter;
            if (colDir.magnitude < collisionSphereRadius)
            {
                float mag = collisionSphereRadius / colDir.magnitude;
                particle.currentForce += colDir * 300 * mag;
            }

            newParticles[id] = particle;
        }
    }

    // Job to integrate particles (Optimized with Burst)
    [BurstCompile]
    private struct IntegrateJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<Particle> particles;
        public NativeArray<Particle> newParticles;
        public float timestep;
        public float particleMass;
        public float boundDamping;
        public Vector3 boxSize;
        public float radius;

        public void Execute(int id)
        {
            Particle particle = particles[id];
            Vector3 vel = particle.velocity + ((particle.currentForce / particleMass) * timestep);
            particle.position += vel * timestep;
            Vector3 topRight = boxSize / 2;
            Vector3 bottomLeft = -boxSize / 2;

            if (particle.position.x - radius < bottomLeft.x)
            {
                vel.x *= boundDamping;
                particle.position.x = bottomLeft.x + radius;
            }

            if (particle.position.y - radius < bottomLeft.y)
            {
                vel.y *= boundDamping;
                particle.position.y = bottomLeft.y + radius;
            }

            if (particle.position.z - radius < bottomLeft.z)
            {
                vel.z *= boundDamping;
                particle.position.z = bottomLeft.z + radius;
            }

            if (particle.position.x + radius > topRight.x)
            {
                vel.x *= boundDamping;
                particle.position.x = topRight.x - radius;
            }

            if (particle.position.y + radius > topRight.y)
            {
                vel.y *= boundDamping;
                particle.position.y = topRight.y - radius;
            }

            if (particle.position.z + radius > topRight.z)
            {
                vel.z *= boundDamping;
                particle.position.z = topRight.z - radius;
            }

            particle.velocity = vel;

            newParticles[id] = particle;
        }
    }
}
