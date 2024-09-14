using System.Collections;
using System.Collections.Generic;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using UnityEngine;

public class SPH_MiroCsharp_Bursted : MonoBehaviour
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

    // NativeArray to store particles for use in jobs 
    private NativeArray<Particle> particles;
    private NativeArray<Particle> newParticles;

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
        // Scheduling jobs to run in parallel
        var densityJob = new ComputeDensityPressureJob
        {
            particles = particles,
            newParticles = newParticles,
            particleMass = particleMass,
            gasConstant = gasConstant,
            restingDensity = restingDensity,
            radius2 = radius2
        };
        JobHandle densityHandle = densityJob.Schedule(totalParticles, 64);

        var forceJob = new ComputeForcesJob
        {
            particles = particles,
            newParticles = newParticles,
            timestep = timestep,
            particleMass = particleMass,
            viscosity = viscosity,
            radius = radius,
            collisionSphereCenter = collisionSphere.position,
            collisionSphereRadius = collisionSphere.localScale.x / 2
        };
        JobHandle forceHandle = forceJob.Schedule(totalParticles, 64, densityHandle);

        var integrationJob = new IntegrateJob
        {
            particles = particles,
            newParticles = newParticles,
            timestep = timestep,
            particleMass = particleMass,
            boundDamping = boundDamping,
            topRight = boxSize / 2,
            bottomLeft = -boxSize / 2,
            radius = radius,
            boxSize = boxSize
        };
        JobHandle integrationHandle = integrationJob.Schedule(totalParticles, 64, forceHandle);

        integrationHandle.Complete(); // Ensure all jobs are finished before moving on

        var temp = particles;
        particles = newParticles;
        newParticles = temp;
    }

    private void Awake()
    {
        particles = new NativeArray<Particle>(totalParticles, Allocator.Persistent); // Initialize NativeArray (Optimized)
        newParticles = new NativeArray<Particle>(totalParticles, Allocator.Persistent); // Initialize NativeArray (Optimized)
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

                    spawnPos += Random.insideUnitSphere * particleRadius * spawnJitter;

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

            for (int i = 0; i < particles.Length; i++)
            {
                Vector3 diff = origin - particles[i].position;
                float distanceSquared = Vector3.Dot(diff, diff);

                if (radius2 > distanceSquared)
                {
                    sum += StdKernel(distanceSquared);
                }
            }

            particle.density = sum * particleMass;
            particle.pressure = gasConstant * ( particle.density - restingDensity );

            newParticles[id] = particle;
        }
    }


    [BurstCompile]
    private struct ComputeForcesJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<Particle> particles;
        public NativeArray<Particle> newParticles;

        public float timestep;
        public float particleMass;
        public float viscosity;
        public float radius;
        public Vector3 collisionSphereCenter;
        public float collisionSphereRadius;

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
            float density2 = particle.density * particles[id].density;
            Vector3 pressure = Vector3.zero;
            Vector3 visc = Vector3.zero;
            float mass2 = particleMass * particleMass;

            for (int i = 0; i < particles.Length; i++)
            {
                if (i == id) continue;

                float dist = Vector3.Distance(particles[i].position, origin);
                if (dist < radius * 2)
                {
                    Vector3 pressureGradientDirection = Vector3.Normalize(particle.position - particle.position);
                    pressure += mass2 * (particle.pressure / density2 + particle.pressure / (particle.density * particle.density)) * SpikyKernelGradient(dist, pressureGradientDirection);
                    visc += viscosity * mass2 * (particle.velocity - particle.velocity) / particle.density * SpikyKernelSecondDerivative(dist);
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


    [BurstCompile]
    private struct IntegrateJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<Particle> particles;
        public NativeArray<Particle> newParticles;
        
        public float timestep;
        public float particleMass;
        public float boundDamping;
        public Vector3 topRight;
        public Vector3 bottomLeft;
        public float radius;
        public Vector3 boxSize;

        public void Execute(int id)
        {
            Particle particle = particles[id];
            
            Vector3 topRight = boxSize / 2;
            Vector3 bottomLeft = -boxSize / 2;

            Vector3 vel = particle.velocity + ((particle.currentForce / particleMass) * timestep);
            particle.position += vel * timestep;

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
