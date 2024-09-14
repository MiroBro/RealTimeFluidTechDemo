using System.Runtime.InteropServices;
using UnityEngine;
using System.Collections.Generic;
[System.Serializable]
[StructLayout(LayoutKind.Sequential, Size = 44)]

public class SPH_Compute_Csharp : MonoBehaviour
{
    [Header("General")]
    public bool showSpheres = false;
    public Vector3Int numToSpawn = new Vector3Int(10, 10, 10);
    public Vector3 boxSize = new Vector3(4, 10, 3);
    public Vector3 spawnBoxCenter = new Vector3(0, 3, 0);
    public Vector3 spawnBox = new Vector3(4, 2, 1.5f);
    public float particleRadius = 0.1f;

    [Header("Particle Rendering")]
    public Mesh particleMesh;
    public float particleRenderSize = 16f;
    public Material material;

    private static readonly int SizeProperty = Shader.PropertyToID("_size");
    private static readonly int ParticlesBufferProperty = Shader.PropertyToID("_particlesBuffer");

    [Header("Fluid Constants")]
    public float boundDamping = -0.3f;
    public float viscosity = -0.003f;
    public float particleMass = 1f;
    public float gasConstant = 2f;
    public float restingDensity = 1f;

    [Header("Time")]
    public float timestep = 0.0001f;
    public Transform sphere;


    [Header("Compute")]
    //public ComputeShader csharpShaderReplacement;
    public SPH_Compute_Shader_Repolacement_But_CSharp_Only csharpShaderReplacement;
    public Particle[] particles;

    //private ComputeBuffer _argsBuffer;
    //public ComputeBuffer _particlesBuffer;

    private int num = 0;

    private void Awake()
    {
        // Spawn Particles
        SpawnParticlesInBox();

        uint[] args = {
            particleMesh.GetIndexCount(0),
            (uint) num,
            particleMesh.GetIndexStart(0),
            particleMesh.GetBaseVertex(0),
            0
        };
        //_argsBuffer = new ComputeBuffer(1, args.Length * sizeof(uint), ComputeBufferType.IndirectArguments);
        //_argsBuffer.SetData(args);

        //InitializeComputeBuffers();
    }

    //private int densityPressureKernel;
    //private int computeForceKernel;

    //private int integrateKernel;

    /*
    private void InitializeComputeBuffers()
    {
        _particlesBuffer = new ComputeBuffer(num, 44);
        _particlesBuffer.SetData(particles);

        densityPressureKernel = csharpShaderReplacement.FindKernel("ComputeDensityPressure");
        computeForceKernel = csharpShaderReplacement.FindKernel("ComputeForces");
        integrateKernel = csharpShaderReplacement.FindKernel("Integrate");

        csharpShaderReplacement.SetInt("particleLength", num);
        csharpShaderReplacement.SetFloat("particleMass", particleMass);
        csharpShaderReplacement.SetFloat("viscosity", viscosity);
        csharpShaderReplacement.SetFloat("gasConstant", gasConstant);
        csharpShaderReplacement.SetFloat("restDensity", restingDensity);
        csharpShaderReplacement.SetFloat("boundDamping", boundDamping);

        csharpShaderReplacement.SetFloat("radius", particleRadius);
        csharpShaderReplacement.SetFloat("radius2", particleRadius * particleRadius);
        csharpShaderReplacement.SetFloat("radius3", particleRadius * particleRadius * particleRadius);
        csharpShaderReplacement.SetFloat("radius4", particleRadius * particleRadius * particleRadius * particleRadius);
        csharpShaderReplacement.SetFloat("radius5", particleRadius * particleRadius * particleRadius * particleRadius * particleRadius);

        csharpShaderReplacement.SetFloat("pi", Mathf.PI);
        csharpShaderReplacement.SetFloat("densityWeightConstant", 0.00497359197162172924277761760539f);
        csharpShaderReplacement.SetFloat("spikyGradient", -0.09947183943243458485555235210782f);
        csharpShaderReplacement.SetFloat("viscLaplacian", 0.39788735772973833942220940843129f);


        csharpShaderReplacement.SetVector("boxSize", boxSize);

        csharpShaderReplacement.SetBuffer(densityPressureKernel, "_particles", _particlesBuffer);
        csharpShaderReplacement.SetBuffer(computeForceKernel, "_particles", _particlesBuffer);
        csharpShaderReplacement.SetBuffer(integrateKernel, "_particles", _particlesBuffer);

    }
    */

    private void SpawnParticlesInBox()
    {
        Vector3 spawnTopLeft = spawnBoxCenter - spawnBox / 2;
        List<Particle> _particles = new List<Particle>();

        for (int x = 0; x < numToSpawn.x; x++)
        {
            for (int y = 0; y < numToSpawn.y; y++)
            {
                for (int z = 0; z < numToSpawn.z; z++)
                {
                    Vector3 spawnPosition = spawnTopLeft + new Vector3(x * particleRadius * 2, y * particleRadius * 2, z * particleRadius * 2) + Random.onUnitSphere * particleRadius * 0.1f;
                    Particle p = new Particle
                    {
                        position = spawnPosition
                    };

                    _particles.Add(p);
                }
            }
        }

        num = _particles.Count;
        particles = _particles.ToArray();
    }

    private void FixedUpdate()
    {

        //csharpShaderReplacement.SetVector("boxSize", boxSize);
        //csharpShaderReplacement.SetFloat("timestep", timestep);
        //csharpShaderReplacement.SetVector("spherePos", sphere.transform.position);
        //csharpShaderReplacement.SetFloat("sphereRadius", sphere.transform.localScale.x / 2);

        //csharpShaderReplacement.Dispatch(densityPressureKernel, num / 100, 1, 1);
        //csharpShaderReplacement.Dispatch(computeForceKernel, num / 100, 1, 1);
        //csharpShaderReplacement.Dispatch(integrateKernel, num / 100, 1, 1);

        // Perform the simulation in C#
        ComputeDensityPressure();  // Density and pressure calculations
        ComputeForces();           // Forces (pressure, viscosity, etc.)
        Integrate();               // Integrate position and velocity changes

        material.SetFloat(SizeProperty, particleRenderSize);
        //material.SetBuffer(ParticlesBufferProperty, _particlesBuffer);

        // Manually render particles since we're not using buffers
        DrawParticles();
    }

    void ComputeDensityPressure()
    {
        for (int i = 0; i < particles.Length; i++)
        {
            particles[i].density = 0f;
            particles[i].pressure = 0f;

            for (int j = 0; j < particles.Length; j++)
            {
                if (i == j) continue;

                Vector3 diff = particles[i].position - particles[j].position;
                float distSquared = diff.sqrMagnitude;

                if (distSquared < particleRadius * particleRadius)
                {
                    float distance = Mathf.Sqrt(distSquared);
                    float q = 1.0f - (distance / particleRadius);
                    particles[i].density += particleMass * q * q * q;  // SPH density kernel function
                }
            }

            // Use the equation of state to calculate pressure
            particles[i].pressure = gasConstant * (particles[i].density - restingDensity);
        }
    }

    void ComputeForces()
    {
        for (int i = 0; i < particles.Length; i++)
        {
            Vector3 pressureForce = Vector3.zero;
            Vector3 viscosityForce = Vector3.zero;

            for (int j = 0; j < particles.Length; j++)
            {
                if (i == j) continue;

                Vector3 diff = particles[i].position - particles[j].position;
                float distSquared = diff.sqrMagnitude;

                if (distSquared < particleRadius * particleRadius)
                {
                    float distance = Mathf.Sqrt(distSquared);
                    Vector3 direction = diff.normalized;
                    float q = 1.0f - (distance / particleRadius);

                    // Pressure force
                    pressureForce += direction * particleMass * (particles[i].pressure + particles[j].pressure) / (2.0f * particles[j].density);

                    // Viscosity force
                    viscosityForce += viscosity * (particles[j].velocity - particles[i].velocity) * q;
                }
            }

            particles[i].currentForce = -pressureForce + viscosityForce;
        }
    }

    void Integrate()
    {
        for (int i = 0; i < particles.Length; i++)
        {
            Particle p = particles[i];

            // Simple Euler integration
            p.velocity += timestep * p.currentForce / p.density;
            p.position += timestep * p.velocity;

            // Apply boundary conditions
            if (p.position.x < -boxSize.x / 2 || p.position.x > boxSize.x / 2)
                p.velocity.x *= boundDamping;
            if (p.position.y < 0 || p.position.y > boxSize.y)
                p.velocity.y *= boundDamping;
            if (p.position.z < -boxSize.z / 2 || p.position.z > boxSize.z / 2)
                p.velocity.z *= boundDamping;

            // Update particle
            particles[i] = p;
        }
    }

    void DrawParticles()
    {
        foreach (var p in particles)
        {
            Matrix4x4 matrix = Matrix4x4.TRS(p.position, Quaternion.identity, Vector3.one * particleRenderSize);
            Graphics.DrawMesh(particleMesh, matrix, material, 0);
        }
    }

    /*
    private void Update()
    {
        if (showSpheres)
        {
            // Manually draw each particle
            DrawParticles();
        }
    }
    */

    /*
    private void Update()
    {
        if (showSpheres) Graphics.DrawMeshInstancedIndirect(particleMesh, 0, material, new Bounds(Vector3.zero, boxSize), _argsBuffer, castShadows: UnityEngine.Rendering.ShadowCastingMode.Off);
    }

    */

    private void OnDrawGizmos()
    {
        Gizmos.color = Color.blue;
        Gizmos.DrawWireCube(Vector3.zero, boxSize);

        if (!Application.isPlaying)
        {
            Gizmos.color = Color.cyan;
            Gizmos.DrawWireCube(spawnBoxCenter, spawnBox);
        }
    }
}
