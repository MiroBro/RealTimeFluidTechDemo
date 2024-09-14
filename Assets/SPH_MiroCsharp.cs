using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SPH_MiroCsharp : MonoBehaviour
{
    [Header("General")]
    public Transform collisionSphere;
    public bool showSpheres = true;
    public Vector3Int numToSpawn = new Vector3Int(10, 10, 10);
    private int totalParticles
    {
        get
        {
            return numToSpawn.x * numToSpawn.y * numToSpawn.z;
        }
    }

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

    //Private variables
    private Particle[] particles;

    private struct Particle
    {
        public float pressure;
        public float density;
        public Vector3 currentForce;
        public Vector3 velocity;
        public Vector3 position;
    }

    private static readonly int SizeProperty = Shader.PropertyToID("_size");
    private static readonly int ParticlesBufferProperty = Shader.PropertyToID("_particlesBuffer");
    private List<Matrix4x4> matrices;

    private void Update()
    {
        if (showSpheres)
        {
            RenderParticles();
        }
    }

    private void FixedUpdate()
    {
        for (int i = 0; i < totalParticles; i++)
        {
            ComputeDensityPressure(i);
        }

        for (int i = 0; i < totalParticles; i++)
        {
            ComputeForces(i);
        }

        for (int i = 0; i < totalParticles; i++)
        {
            Integrate(i);
        }
    }

    private void Awake()
    {
        // Initialize particles and matrices
        SpawnParticlesInBox();
        InitializeMatrices();
    }

    private void InitializeParticles()
    {
        for (int i = 0; i < totalParticles; i++)
        {
            particles[i].position = spawnCenter + new Vector3(
                (i % numToSpawn.x) * particleRadius * 2,
                ((i / numToSpawn.x) % numToSpawn.y) * particleRadius * 2,
                (i / (numToSpawn.x * numToSpawn.x)) * particleRadius * 2
            );
            particles[i].velocity = Vector3.zero;
            particles[i].currentForce = Vector3.zero;
            particles[i].density = restingDensity;
            particles[i].pressure = 0;
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

    private void ComputeDensityPressure(int id)
    {
        Vector3 origin = particles[id].position;
        float sum = 0;

        for (int i = 0; i < totalParticles; i++)
        {
            Vector3 diff = origin - particles[i].position;
            float distanceSquared = Vector3.Dot(diff, diff);

            if (particleRadius * particleRadius > distanceSquared)
            {
                float x = (particleRadius * particleRadius) - distanceSquared;
                sum += StdKernel(distanceSquared);
            }
        }

        particles[id].density = sum * particleMass;
        particles[id].pressure = gasConstant * (particles[id].density - restingDensity);

        if (particles[id].pressure <= 0) particles[id].pressure = 0;
    }

    private void ComputeForces(int id)
    {
        Vector3 origin = particles[id].position;
        float density2 = particles[id].density * particles[id].density;
        Vector3 pressure = Vector3.zero;
        Vector3 visc = Vector3.zero;
        float mass2 = particleMass * particleMass;

        for (int i = 0; i < totalParticles; i++)
        {
            if (i == id) continue;

            float dist = Vector3.Distance(particles[i].position, origin);
            if (dist < particleRadius * 2)
            {
                Vector3 pressureGradientDirection = Vector3.Normalize(particles[id].position - particles[i].position);
                pressure += mass2 * (particles[id].pressure / density2 + particles[i].pressure / (particles[i].density * particles[i].density)) * SpikyKernelGradient(dist, pressureGradientDirection);
                visc += viscosity * mass2 * (particles[i].velocity - particles[id].velocity) / particles[i].density * SpikyKernelSecondDerivative(dist);
            }
        }

        particles[id].currentForce = new Vector3(0, (-9.81f * particleMass), 0) - pressure + visc;

        Vector3 colDir = particles[id].position - collisionSphere.position;
        if (colDir.magnitude < collisionSphere.localScale.x / 2)
        {
            float mag = (collisionSphere.localScale.x / 2) / colDir.magnitude;
            particles[id].currentForce += colDir * 300 * mag;
        }
    }

    private void Integrate(int id)
    {
        Vector3 topRight = boxSize / 2;
        Vector3 bottomLeft = -boxSize / 2;
        Vector3 vel = particles[id].velocity + ((particles[id].currentForce / particleMass) * timestep);
        particles[id].position += vel * timestep;

        if (particles[id].position.x - particleRadius < bottomLeft.x)
        {
            vel.x *= boundDamping;
            particles[id].position.x = bottomLeft.x + particleRadius;
        }

        if (particles[id].position.y - particleRadius < bottomLeft.y)
        {
            vel.y *= boundDamping;
            particles[id].position.y = bottomLeft.y + particleRadius;
        }

        if (particles[id].position.z - particleRadius < bottomLeft.z)
        {
            vel.z *= boundDamping;
            particles[id].position.z = bottomLeft.z + particleRadius;
        }

        if (particles[id].position.x + particleRadius > topRight.x)
        {
            vel.x *= boundDamping;
            particles[id].position.x = topRight.x - particleRadius;
        }

        if (particles[id].position.y + particleRadius > topRight.y)
        {
            vel.y *= boundDamping;
            particles[id].position.y = topRight.y - particleRadius;
        }

        if (particles[id].position.z + particleRadius > topRight.z)
        {
            vel.z *= boundDamping;
            particles[id].position.z = topRight.z - particleRadius;
        }

        particles[id].velocity = vel;
    }

    private float StdKernel(float distanceSquared)
    {
        float x = 1.0f - distanceSquared / (particleRadius * particleRadius);
        return 315f / (64f * Mathf.PI * Mathf.Pow(particleRadius, 3)) * x * x * x;
    }

    private float SpikyKernelFirstDerivative(float distance)
    {
        float x = 1.0f - distance / particleRadius;
        return -45.0f / (Mathf.PI * Mathf.Pow(particleRadius, 4)) * x * x;
    }

    private float SpikyKernelSecondDerivative(float distance)
    {
        float x = 1.0f - distance / particleRadius;
        return 90f / (Mathf.PI * Mathf.Pow(particleRadius, 5)) * x;
    }

    private Vector3 SpikyKernelGradient(float distance, Vector3 directionFromCenter)
    {
        return SpikyKernelFirstDerivative(distance) * directionFromCenter;
    }

    private void OnDrawGizmos()
    {
        Gizmos.color = Color.blue;
        Gizmos.DrawWireCube(Vector3.zero, boxSize);
        if (!Application.isPlaying)
        {
            Gizmos.color = Color.cyan;
            Gizmos.DrawWireSphere(spawnCenter, 0.1f);
        }
    }

    private void SpawnParticlesInBox()
    {
        Vector3 spawnPoint = spawnCenter;
        particles = new Particle[totalParticles];

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
}
