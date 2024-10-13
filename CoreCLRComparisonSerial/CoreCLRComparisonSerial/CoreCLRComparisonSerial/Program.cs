using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Numerics;

namespace SPHSimulationCoreCLR
{
    struct Int3 : IEquatable<Int3>
    {
        public int X, Y, Z;

        public Int3(int x, int y, int z)
        {
            X = x;
            Y = y;
            Z = z;
        }

        public override bool Equals(object obj)
        {
            return obj is Int3 other && Equals(other);
        }

        public bool Equals(Int3 other)
        {
            return X == other.X && Y == other.Y && Z == other.Z;
        }

        public override int GetHashCode()
        {
            // Simple hash code combining
            unchecked
            {
                int hash = 17;
                hash = hash * 31 + X;
                hash = hash * 31 + Y;
                hash = hash * 31 + Z;
                return hash;
            }
        }

        public static Int3 operator +(Int3 a, Int3 b)
        {
            return new Int3(a.X + b.X, a.Y + b.Y, a.Z + b.Z);
        }
    }

    struct Particle
    {
        public Vector3 Position;
        public Vector3 Velocity;
        public Vector3 CurrentForce;
        public float Density;
        public float Pressure;
    }

    class SPHSimulation
    {
        // Fluid constants
        private const float ParticleMass = 1f;
        private const float GasConstant = 2f;
        private const float RestingDensity = 1f;
        private const float Viscosity = -0.003f;
        private const float TimeStep = 0.007f;
        private const float BoundDamping = -0.3f;
        private static readonly Vector3 BoxSize = new Vector3(4, 10, 3);
        private const float Radius = 0.1f;
        private static readonly float Radius2 = Radius * Radius;

        private Particle[] particles;
        private Particle[] newParticles;
        private Dictionary<Int3, List<int>> particleGridMap;
        private Int3[] neighborOffsets;

        private Stopwatch stopwatch;
        private long first100TotalTime = 0;
        private long remaining1000TotalTime = 0;

        // New: List to store individual iteration times
        private List<long> iterationTimes;

        static void Main(string[] args)
        {
            SPHSimulation simulation = new SPHSimulation();
            simulation.Initialize();
            simulation.RunSimulation();
            simulation.PrintResults();
        }

        private void Initialize()
        {
            int numberOfParticles = 8000;
            particles = new Particle[numberOfParticles];
            newParticles = new Particle[numberOfParticles];
            neighborOffsets = new Int3[27];
            InitializeNeighborOffsets();
            particleGridMap = new Dictionary<Int3, List<int>>(numberOfParticles);

            // Initialize particles
            SpawnParticlesInBox(numberOfParticles);

            // Initialize stopwatch
            stopwatch = new Stopwatch();

            // Initialize the list to store iteration times
            iterationTimes = new List<long>(1100);
        }

        private void RunSimulation()
        {
            for (int i = 0; i < 1100; i++)
            {
                stopwatch.Restart();

                RunCalculations();

                stopwatch.Stop();

                long elapsedMs = stopwatch.ElapsedMilliseconds;
                iterationTimes.Add(elapsedMs);

                // Record timings for first 100 and remaining 1000 iterations
                if (i < 100)
                {
                    first100TotalTime += elapsedMs;
                }
                else
                {
                    remaining1000TotalTime += elapsedMs;
                }
            }
        }

        private void PrintResults()
        {
            Console.WriteLine($"Non-Bursted SPH: Average time for for the first 100 iterations: {first100TotalTime / 100.0f} ms");
            Console.WriteLine($"Non-Bursted SPH: Average time for remaining 1000 iterations: {remaining1000TotalTime / 1000.0f} ms");
            Console.WriteLine("Individual iteration times (in ms):");

            for (int i = 0; i < iterationTimes.Count; i++)
            {
                Console.WriteLine($"Iteration {i + 1}: {iterationTimes[i]} ms");
            }

            Console.WriteLine("Simulation complete!");
        }

        private void SpawnParticlesInBox(int numParticles)
        {
            Random random = new Random(1234);
            for (int i = 0; i < numParticles; i++)
            {
                float x = (float)random.NextDouble() * BoxSize.X;
                float y = (float)random.NextDouble() * BoxSize.Y;
                float z = (float)random.NextDouble() * BoxSize.Z;

                particles[i] = new Particle
                {
                    Position = new Vector3(x, y, z),
                    Velocity = Vector3.Zero,
                    CurrentForce = Vector3.Zero,
                    Density = RestingDensity,
                    Pressure = 0f
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
                        neighborOffsets[index++] = new Int3(x, y, z);
                    }
                }
            }
        }

        private void RunCalculations()
        {
            particleGridMap.Clear();

            // Populate the particle grid map with current particle positions
            for (int i = 0; i < particles.Length; i++)
            {
                Int3 gridPos = HashPosition(particles[i].Position);
                if (!particleGridMap.ContainsKey(gridPos))
                {
                    particleGridMap[gridPos] = new List<int>();
                }
                particleGridMap[gridPos].Add(i);
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
            var temp = particles;
            particles = newParticles;
            newParticles = temp;
        }

        private Int3 HashPosition(Vector3 position)
        {
            int x = (int)MathF.Floor(position.X / (Radius * 2.5f));
            int y = (int)MathF.Floor(position.Y / (Radius * 2.5f));
            int z = (int)MathF.Floor(position.Z / (Radius * 2.5f));
            return new Int3(x, y, z);
        }

        private void ComputeDensityPressure(int id)
        {
            Particle particle = particles[id];
            Vector3 origin = particle.Position;
            float densitySum = 0f;

            Int3 gridPos = HashPosition(origin);

            foreach (var offset in neighborOffsets)
            {
                Int3 neighborPos = gridPos + offset;
                if (particleGridMap.TryGetValue(neighborPos, out List<int> neighbors))
                {
                    foreach (var neighborIndex in neighbors)
                    {
                        Particle neighbor = particles[neighborIndex];
                        Vector3 diff = origin - neighbor.Position;
                        float distanceSquared = diff.LengthSquared();

                        if (distanceSquared < Radius2)
                        {
                            densitySum += StandardKernel(distanceSquared);
                        }
                    }
                }
            }

            particle.Density = densitySum * ParticleMass;
            particle.Pressure = GasConstant * (particle.Density - RestingDensity);
            newParticles[id] = particle;
        }

        private float StandardKernel(float distanceSquared)
        {
            float x = 1.0f - distanceSquared / Radius2;
            return (315f / (64f * (float)Math.PI * MathF.Pow(Radius2, 1.5f))) * x * x * x;
        }

        private void ComputeForces(int id)
        {
            Particle particle = particles[id];
            Vector3 origin = particle.Position;
            Vector3 pressureForce = Vector3.Zero;
            Vector3 viscousForce = Vector3.Zero;

            Int3 gridPos = HashPosition(origin);

            foreach (var offset in neighborOffsets)
            {
                Int3 neighborPos = gridPos + offset;
                if (particleGridMap.TryGetValue(neighborPos, out List<int> neighbors))
                {
                    foreach (var neighborIndex in neighbors)
                    {
                        if (neighborIndex == id) continue;

                        Particle neighbor = particles[neighborIndex];
                        Vector3 diff = origin - neighbor.Position;
                        float distanceSquared = diff.LengthSquared();

                        if (distanceSquared < Radius2)
                        {
                            float distance = MathF.Sqrt(distanceSquared);
                            if (distance > 0f)
                            {
                                pressureForce += PressureKernelGradient(particle, neighbor, distance, diff);
                                viscousForce += ViscosityKernel(particle, neighbor, distance);
                            }
                        }
                    }
                }
            }

            // External force (gravity)
            Vector3 externalForce = new Vector3(0, -9.81f * ParticleMass, 0);
            particle.CurrentForce = externalForce - pressureForce + viscousForce;
            newParticles[id] = particle;
        }

        private Vector3 PressureKernelGradient(Particle particle, Particle neighbor, float distance, Vector3 direction)
        {
            float gradValue = -45.0f / ((float)Math.PI * MathF.Pow(Radius, 4)) * MathF.Pow(1.0f - distance / Radius, 2);
            return gradValue * direction * (particle.Pressure + neighbor.Pressure) / (2 * neighbor.Density);
        }

        private Vector3 ViscosityKernel(Particle particle, Particle neighbor, float distance)
        {
            float secondDerivative = 90f / ((float)Math.PI * MathF.Pow(Radius, 5)) * (1.0f - distance / Radius);
            return Viscosity * secondDerivative * (neighbor.Velocity - particle.Velocity) / neighbor.Density;
        }

        private void Integrate(int id)
        {
            Particle particle = particles[id];
            particle.Velocity += (particle.CurrentForce / ParticleMass) * TimeStep;
            particle.Position += particle.Velocity * TimeStep;

            // Handle bounding box collisions
            Vector3 min = -BoxSize / 2;
            Vector3 max = BoxSize / 2;

            Vector3 velocity = particle.Velocity;

            // Check and modify the X component of the velocity
            if (particle.Position.X < min.X || particle.Position.X > max.X)
            {
                velocity.X *= BoundDamping;
            }

            // Check and modify the Y component of the velocity
            if (particle.Position.Y < min.Y || particle.Position.Y > max.Y)
            {
                velocity.Y *= BoundDamping;
            }

            // Check and modify the Z component of the velocity
            if (particle.Position.Z < min.Z || particle.Position.Z > max.Z)
            {
                velocity.Z *= BoundDamping;
            }

            // Update the particle's velocity
            particle.Velocity = velocity;

            newParticles[id] = particle;
        }
    }
}
