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

    private NativeParallelMultiHashMap<int3, int> particleGridMap;
    private NativeArray<int3> neighborOffsets;
    private SpatialHash spatialHash;

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

    [BurstCompile]
    public struct SpatialHash
    {
        public float cellSize;

        public int3 Hash(Vector3 position)
        {
            return new int3(
                Mathf.FloorToInt(position.x / cellSize),
                Mathf.FloorToInt(position.y / cellSize),
                Mathf.FloorToInt(position.z / cellSize)
            );
        }

        public int3[] NeighborOffsets()
        {
            return new int3[]
            {
            new int3( 0, 0, 0), new int3( 0, 0, 1), new int3( 0, 0,-1),
            new int3( 0, 1, 0), new int3( 0, 1, 1), new int3( 0, 1,-1),
            new int3( 0,-1, 0), new int3( 0,-1, 1), new int3( 0,-1,-1),
            new int3( 1, 0, 0), new int3( 1, 0, 1), new int3( 1, 0,-1),
            new int3( 1, 1, 0), new int3( 1, 1, 1), new int3( 1, 1,-1),
            new int3( 1,-1, 0), new int3( 1,-1, 1), new int3( 1,-1,-1),
            new int3(-1, 0, 0), new int3(-1, 0, 1), new int3(-1, 0,-1),
            new int3(-1, 1, 0), new int3(-1, 1, 1), new int3(-1, 1,-1),
            new int3(-1,-1, 0), new int3(-1,-1, 1), new int3(-1,-1,-1)
            };
        }
    }


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
        // Ensure particleGridMap is initialized before clearing
        if (!particleGridMap.IsCreated)
        {
            Debug.LogError("particleGridMap has not been initialized.");
            return;
        }

        // Clear particleGridMap before use
        particleGridMap.Clear();

        // Populate particleGridMap with current particle positions
        for (int i = 0; i < totalParticles; i++)
        {
            int3 gridPos = spatialHash.Hash(particles[i].position);
            particleGridMap.Add(gridPos, i);
        }

        // Scheduling jobs as before
        var densityJob = new ComputeDensityPressureJob
        {
            particles = particles,
            newParticles = newParticles,
            particleMass = particleMass,
            gasConstant = gasConstant,
            restingDensity = restingDensity,
            radius2 = radius2,
            spatialHash = spatialHash,
            particleGridMap = particleGridMap,  // Pass the map
            neighborOffsets = neighborOffsets  // Pass neighbor offsets
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
            spatialHash = spatialHash,
            particleGridMap = particleGridMap,  // Pass the map
            neighborOffsets = neighborOffsets  // Pass neighbor offsets
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

        // Swap particle buffers for the next frame
        var temp = particles;
        particles = newParticles;
        newParticles = temp;
    }


    private void Awake()
    {
        // Ensure the particle count is valid
        int particleCount = totalParticles;
        if (particleCount <= 0)
        {
            Debug.LogError("Total particles should be greater than zero.");
            return;
        }

        // Allocate the particle arrays
        particles = new NativeArray<Particle>(particleCount, Allocator.Persistent);
        newParticles = new NativeArray<Particle>(particleCount, Allocator.Persistent);

        // Initialize the particleGridMap with enough capacity
        particleGridMap = new NativeParallelMultiHashMap<int3, int>(particleCount, Allocator.Persistent);

        // Initialize the spatial hash with proper cell size
        spatialHash = new SpatialHash { cellSize = particleRadius * 2.5f };

        // Initialize the neighbor offsets as a NativeArray
        neighborOffsets = new NativeArray<int3>(spatialHash.NeighborOffsets().Length, Allocator.Persistent);
        neighborOffsets.CopyFrom(spatialHash.NeighborOffsets());

        // Initialize particles and matrices
        SpawnParticlesInBox();
        InitializeMatrices();
    }

    private void OnDestroy()
    {
        if (particles.IsCreated) particles.Dispose();
        if (newParticles.IsCreated) newParticles.Dispose();
        if (particleGridMap.IsCreated) particleGridMap.Dispose();
        if (neighborOffsets.IsCreated) neighborOffsets.Dispose();
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

    [BurstCompile]// (FloatMode = FloatMode.Fast)]
    private struct ComputeDensityPressureJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<Particle> particles;
        public NativeArray<Particle> newParticles;
        public float particleMass;
        public float gasConstant;
        public float restingDensity;
        public float radius2;
        public SpatialHash spatialHash;

        [ReadOnly] public NativeParallelMultiHashMap<int3, int> particleGridMap; // Mark as ReadOnly
        [ReadOnly] public NativeArray<int3> neighborOffsets;  // Already ReadOnly


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

            int3 gridPos = spatialHash.Hash(origin);

            for (int i = 0; i < neighborOffsets.Length; i++)
            {
                int3 neighborPos = gridPos + neighborOffsets[i];
                if (particleGridMap.TryGetFirstValue(neighborPos, out int neighborIndex, out var it))
                {
                    do
                    {
                        Particle neighbor = particles[neighborIndex];
                        Vector3 diff = origin - neighbor.position;
                        float distanceSquared = Vector3.Dot(diff, diff);

                        if (radius2 > distanceSquared)
                        {
                            sum += StdKernel(distanceSquared);
                        }
                    }
                    while (particleGridMap.TryGetNextValue(out neighborIndex, ref it));
                }
            }

            particle.density = sum * particleMass;
            particle.pressure = gasConstant * (particle.density - restingDensity);

            newParticles[id] = particle;
        }
    }

    [BurstCompile]// (FloatMode = FloatMode.Fast)]
    private struct ComputeForcesJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<Particle> particles;
        public NativeArray<Particle> newParticles;
        public float particleMass;
        public float viscosity;
        public float radius;
        public Vector3 collisionSphereCenter;
        public float collisionSphereRadius;
        public SpatialHash spatialHash;

        [ReadOnly] public NativeParallelMultiHashMap<int3, int> particleGridMap; // Mark as ReadOnly
        [ReadOnly] public NativeArray<int3> neighborOffsets;  // Already ReadOnly

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

            int3 gridPos = spatialHash.Hash(origin);

            for (int i = 0; i < neighborOffsets.Length; i++)
            {
                int3 neighborPos = gridPos + neighborOffsets[i];
                if (particleGridMap.TryGetFirstValue(neighborPos, out int neighborIndex, out var it))
                {
                    do
                    {
                        if (neighborIndex == id) continue;

                        Particle neighbor = particles[neighborIndex];
                        Vector3 diff = origin - neighbor.position;

                        float distSquared = diff.sqrMagnitude;
                        if (distSquared < radius * radius)
                        {
                            float distance = Mathf.Sqrt(distSquared);
                            Vector3 direction = diff.normalized;

                            Vector3 pressureGradientDirection = Vector3.Normalize(particle.position - neighbor.position);
                            pressure += mass2 * (particle.pressure / density2 + neighbor.pressure / (neighbor.density * neighbor.density)) * SpikyKernelGradient(distSquared, pressureGradientDirection);
                            visc += viscosity * mass2 * (neighbor.velocity - particle.velocity) / neighbor.density * SpikyKernelSecondDerivative(distance);
                        }
                    }
                    while (particleGridMap.TryGetNextValue(out neighborIndex, ref it));
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

    [BurstCompile]// (FloatMode = FloatMode.Fast)]
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
