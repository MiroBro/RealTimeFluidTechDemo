using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Threading.Tasks;
using System.Buffers;

namespace FluidSimulationCoreCLR
{
    public struct Vector3Int
    {
        public int X;
        public int Y;
        public int Z;

        public Vector3Int(int x, int y, int z)
        {
            X = x;
            Y = y;
            Z = z;
        }

        public static Vector3Int operator +(Vector3Int a, Vector3Int b)
        {
            return new Vector3Int(a.X + b.X, a.Y + b.Y, a.Z + b.Z);
        }

        public static Vector3Int operator -(Vector3Int a, Vector3Int b)
        {
            return new Vector3Int(a.X - b.X, a.Y - b.Y, a.Z - b.Z);
        }

        public override bool Equals(object obj)
        {
            if (!(obj is Vector3Int)) return false;
            Vector3Int other = (Vector3Int)obj;
            return X == other.X && Y == other.Y && Z == other.Z;
        }

        public override int GetHashCode()
        {
            return HashCode.Combine(X, Y, Z);
        }

        public override string ToString()
        {
            return $"({X}, {Y}, {Z})";
        }
    }

    public class FluidSimulation
    {
        public Vector3Int NumToSpawn;
        public int TotalParticles => NumToSpawn.X * NumToSpawn.Y * NumToSpawn.Z;
        public Vector3 BoxSize = new Vector3(4, 10, 3);
        public Vector3 SpawnCenter;
        public float ParticleRadius = 0.1f;
        public float SpawnJitter = 0.2f;

        public float BoundDamping = -0.3f;
        public float Viscosity = -0.003f;
        public float ParticleMass = 1f;
        public float GasConstant = 2f;
        public float RestingDensity = 1f;
        public float Timestep = 0.007f;
        public float Radius => ParticleRadius;
        public float Radius2 => ParticleRadius * ParticleRadius;

        private Particle[] particles;
        private Particle[] newParticles;
        private Dictionary<Vector3Int, List<int>> particleGridMap;
        private Vector3Int[] neighborOffsets;
        private SpatialHash spatialHash;
        private readonly ArrayPool<Particle> _particlePool = ArrayPool<Particle>.Shared;

        private struct Particle
        {
            public float Pressure;
            public float Density;
            public Vector3 CurrentForce;
            public Vector3 Velocity;
            public Vector3 Position;
        }

        public struct SpatialHash
        {
            public float CellSize;

            public Vector3Int Hash(Vector3 position)
            {
                return new Vector3Int(
                    (int)Math.Floor(position.X / CellSize),
                    (int)Math.Floor(position.Y / CellSize),
                    (int)Math.Floor(position.Z / CellSize)
                );
            }

            public Vector3Int[] NeighborOffsets()
            {
                return new Vector3Int[]
                {
                    new Vector3Int( 0, 0, 0), new Vector3Int( 0, 0, 1), new Vector3Int( 0, 0,-1),
                    new Vector3Int( 0, 1, 0), new Vector3Int( 0, 1, 1), new Vector3Int( 0, 1,-1),
                    new Vector3Int( 0,-1, 0), new Vector3Int( 0,-1, 1), new Vector3Int( 0,-1,-1),
                    new Vector3Int( 1, 0, 0), new Vector3Int( 1, 0, 1), new Vector3Int( 1, 0,-1),
                    new Vector3Int( 1, 1, 0), new Vector3Int( 1, 1, 1), new Vector3Int( 1, 1,-1),
                    new Vector3Int( 1,-1, 0), new Vector3Int( 1,-1, 1), new Vector3Int( 1,-1,-1),
                    new Vector3Int(-1, 0, 0), new Vector3Int(-1, 0, 1), new Vector3Int(-1, 0,-1),
                    new Vector3Int(-1, 1, 0), new Vector3Int(-1, 1, 1), new Vector3Int(-1, 1,-1),
                    new Vector3Int(-1,-1, 0), new Vector3Int(-1,-1, 1), new Vector3Int(-1,-1,-1)
                };
            }
        }

        public FluidSimulation(Vector3Int amountToSpawn)
        {
            NumToSpawn = amountToSpawn;
            particles = _particlePool.Rent(TotalParticles);
            newParticles = _particlePool.Rent(TotalParticles);

            particleGridMap = new Dictionary<Vector3Int, List<int>>(TotalParticles);
            spatialHash = new SpatialHash { CellSize = ParticleRadius * 2.5f };
            neighborOffsets = spatialHash.NeighborOffsets();

            SpawnParticlesInBox();
        }

        public void Update()
        {
            particleGridMap.Clear();

            Parallel.For(0, TotalParticles, i =>
            {
                Vector3Int gridPos = spatialHash.Hash(particles[i].Position);
                lock (particleGridMap)
                {
                    if (!particleGridMap.ContainsKey(gridPos))
                        particleGridMap[gridPos] = new List<int>();
                    particleGridMap[gridPos].Add(i);
                }
            });

            // Parallelize density, pressure, and force calculation
            Parallel.For(0, TotalParticles, i =>
            {
                ComputeDensityPressure(i);
                ComputeForces(i);
                Integrate(i);
            });

            var temp = particles;
            particles = newParticles;
            newParticles = temp;
        }

        private void ComputeDensityPressure(int id)
        {
            Particle particle = particles[id];
            Vector3 origin = particle.Position;
            float sum = 0;
            Vector3Int gridPos = spatialHash.Hash(origin);

            foreach (var offset in neighborOffsets)
            {
                Vector3Int neighborPos = gridPos + offset;
                if (particleGridMap.TryGetValue(neighborPos, out List<int> neighbors))
                {
                    foreach (int neighborIndex in neighbors)
                    {
                        Particle neighbor = particles[neighborIndex];
                        Vector3 diff = origin - neighbor.Position;
                        float distanceSquared = Vector3.Dot(diff, diff);

                        if (Radius2 > distanceSquared)
                        {
                            sum += StdKernel(distanceSquared);
                        }
                    }
                }
            }

            particle.Density = sum * ParticleMass;
            particle.Pressure = GasConstant * (particle.Density - RestingDensity);
            newParticles[id] = particle;
        }

        private void ComputeForces(int id)
        {
            Particle particle = particles[id];
            Vector3 origin = particle.Position;
            float density2 = particle.Density * particle.Density;
            Vector3 pressure = Vector3.Zero;
            Vector3 visc = Vector3.Zero;
            float mass2 = ParticleMass * ParticleMass;

            Vector3Int gridPos = spatialHash.Hash(origin);

            foreach (var offset in neighborOffsets)
            {
                Vector3Int neighborPos = gridPos + offset;
                if (particleGridMap.TryGetValue(neighborPos, out List<int> neighbors))
                {
                    foreach (int neighborIndex in neighbors)
                    {
                        if (neighborIndex == id) continue;

                        Particle neighbor = particles[neighborIndex];
                        Vector3 diff = origin - neighbor.Position;
                        float distSquared = diff.LengthSquared();

                        if (distSquared < Radius * Radius)
                        {
                            float distance = (float)Math.Sqrt(distSquared);
                            Vector3 direction = Vector3.Normalize(diff);

                            pressure += mass2 * (particle.Pressure / density2 + neighbor.Pressure / (neighbor.Density * neighbor.Density)) * SpikyKernelGradient(distSquared, direction);
                            visc += Viscosity * mass2 * (neighbor.Velocity - particle.Velocity) / neighbor.Density * SpikyKernelSecondDerivative(distance);
                        }
                    }
                }
            }

            particle.CurrentForce = new Vector3(0, -9.81f * ParticleMass, 0) - pressure + visc;
            newParticles[id] = particle;
        }

        private void Integrate(int id)
        {
            Particle particle = particles[id];
            Vector3 vel = particle.Velocity + ((particle.CurrentForce / ParticleMass) * Timestep);
            particle.Position += vel * Timestep;

            Vector3 topRight = BoxSize / 2;
            Vector3 bottomLeft = -BoxSize / 2;

            for (int i = 0; i < 3; i++)
            {
                if (particle.Position[i] < bottomLeft[i])
                {
                    particle.Position[i] = bottomLeft[i];
                    vel[i] *= BoundDamping;
                }
                if (particle.Position[i] > topRight[i])
                {
                    particle.Position[i] = topRight[i];
                    vel[i] *= BoundDamping;
                }
            }

            particle.Velocity = vel;
            newParticles[id] = particle;
        }

        private float StdKernel(float distanceSquared)
        {
            float x = Radius2 - distanceSquared;
            return 315f / (64f * (float)Math.PI * MathF.Pow(Radius, 9)) * MathF.Pow(x, 3);
        }

        private Vector3 SpikyKernelGradient(float distSquared, Vector3 direction)
        {
            float x = Radius2 - distSquared;
            return -45f / ((float)Math.PI * MathF.Pow(Radius, 6)) * MathF.Pow(x, 2) * direction;
        }

        private float SpikyKernelSecondDerivative(float distance)
        {
            float x = Radius - distance;
            return 45f / ((float)Math.PI * MathF.Pow(Radius, 6)) * x;
        }

        private void SpawnParticlesInBox()
        {
            int index = 0;
            Random random = new Random();
            for (int x = 0; x < NumToSpawn.X; x++)
            {
                for (int y = 0; y < NumToSpawn.Y; y++)
                {
                    for (int z = 0; z < NumToSpawn.Z; z++)
                    {
                        Vector3 pos = new Vector3(x, y, z) / new Vector3(NumToSpawn.X, NumToSpawn.Y, NumToSpawn.Z) * BoxSize;
                        pos += SpawnCenter;
                        pos += new Vector3(
                            (float)(random.NextDouble() * 2 - 1) * SpawnJitter,
                            (float)(random.NextDouble() * 2 - 1) * SpawnJitter,
                            (float)(random.NextDouble() * 2 - 1) * SpawnJitter
                        );


                        particles[index].Position = pos;
                        particles[index].Velocity = Vector3.Zero;
                        particles[index].Density = 0;
                        particles[index].Pressure = 0;
                        index++;
                    }
                }
            }
        }
    }
}
