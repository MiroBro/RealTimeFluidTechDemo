using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FluidSimulationCoreCLR
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Numerics;
    using System.Threading.Tasks;

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
        public Vector3Int NumToSpawn; //= new Vector3Int(10, 10, 10);
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

        // Particle array
        private Particle[] particles;
        private Particle[] newParticles;

        private Dictionary<Vector3Int, List<int>> particleGridMap;
        private Vector3Int[] neighborOffsets;
        private SpatialHash spatialHash;

        // Particle struct
        private struct Particle
        {
            public float Pressure;
            public float Density;
            public Vector3 CurrentForce;
            public Vector3 Velocity;
            public Vector3 Position;
        }

        // Spatial hash for finding neighboring particles
        public struct SpatialHash
        {
            public float CellSize;

            public Vector3Int Hash(Vector3 position)
            {
                Vector3Int gridPos = new Vector3Int(
                    (int)Math.Floor(position.X / CellSize),
                    (int)Math.Floor(position.Y / CellSize),
                    (int)Math.Floor(position.Z / CellSize)
                );

                // Check if the grid position is valid
                if (gridPos.X < -100 || gridPos.X > 100 ||
                    gridPos.Y < -100 || gridPos.Y > 100 ||
                    gridPos.Z < -100 || gridPos.Z > 100)
                {
                    throw new Exception($"Particle position is out of bounds: {position}, Grid: {gridPos}");
                }

                return gridPos;
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
            particles = new Particle[TotalParticles];
            newParticles = new Particle[TotalParticles];

            particleGridMap = new Dictionary<Vector3Int, List<int>>(TotalParticles);
            spatialHash = new SpatialHash { CellSize = ParticleRadius * 2.5f };
            neighborOffsets = spatialHash.NeighborOffsets();

            SpawnParticlesInBox();
        }

        public void Update()
        {
            particleGridMap.Clear();

            // Populate the particle grid
            for (int i = 0; i < TotalParticles; i++)
            {
                Vector3Int gridPos = spatialHash.Hash(particles[i].Position);
                if (!particleGridMap.ContainsKey(gridPos))
                {
                    particleGridMap[gridPos] = new List<int>();
                }
                particleGridMap[gridPos].Add(i);
            }

            // Run jobs in parallel
            Parallel.For(0, TotalParticles, i =>
            {
                ComputeDensityPressure(i);
            });

            Parallel.For(0, TotalParticles, i =>
            {
                ComputeForces(i);
            });

            Parallel.For(0, TotalParticles, i =>
            {
                Integrate(i);
            });

            // Swap buffers
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

            // Boundary conditions
            if (particle.Position.X - Radius < bottomLeft.X)
            {
                vel.X *= BoundDamping;
                particle.Position.X = bottomLeft.X + Radius;
            }

            if (particle.Position.Y - Radius < bottomLeft.Y)
            {
                vel.Y *= BoundDamping;
                particle.Position.Y = bottomLeft.Y + Radius;
            }

            if (particle.Position.Z - Radius < bottomLeft.Z)
            {
                vel.Z *= BoundDamping;
                particle.Position.Z = bottomLeft.Z + Radius;
            }

            if (particle.Position.X + Radius > topRight.X)
            {
                vel.X *= BoundDamping;
                particle.Position.X = topRight.X - Radius;
            }

            if (particle.Position.Y + Radius > topRight.Y)
            {
                vel.Y *= BoundDamping;
                particle.Position.Y = topRight.Y - Radius;
            }

            if (particle.Position.Z + Radius > topRight.Z)
            {
                vel.Z *= BoundDamping;
                particle.Position.Z = topRight.Z - Radius;
            }

            particle.Velocity = vel;
            newParticles[id] = particle;
        }

        private void SpawnParticlesInBox()
        {
            Random rand = new Random();

            for (int z = 0; z < NumToSpawn.Z; z++)
            {
                for (int y = 0; y < NumToSpawn.Y; y++)
                {
                    for (int x = 0; x < NumToSpawn.X; x++)
                    {
                        int index = z * NumToSpawn.Y * NumToSpawn.X + y * NumToSpawn.X + x;
                        Particle particle = new Particle();

                        Vector3 position = new Vector3(
                            ((float)x / (float)NumToSpawn.X) * BoxSize.X,
                            ((float)y / (float)NumToSpawn.Y) * BoxSize.Y,
                            ((float)z / (float)NumToSpawn.Z) * BoxSize.Z
                        );

                        Vector3 jitter = new Vector3(
                            (float)(rand.NextDouble() * SpawnJitter - SpawnJitter / 2),
                            (float)(rand.NextDouble() * SpawnJitter - SpawnJitter / 2),
                            (float)(rand.NextDouble() * SpawnJitter - SpawnJitter / 2)
                        );

                        particle.Position = position + jitter;
                        particle.Velocity = Vector3.Zero;

                        particles[index] = particle;
                    }
                }
            }
        }

        private float StdKernel(float r2)
        {
            float r = (float)Math.Sqrt(r2);
            float h = Radius;
            return 315f / (64f * (float)Math.PI * (float)Math.Pow(h, 9)) * (float)Math.Pow(h * h - r * r, 3);
        }

        private Vector3 SpikyKernelGradient(float r2, Vector3 r)
        {
            float rMagnitude = (float)Math.Sqrt(r2);
            if (rMagnitude == 0) return Vector3.Zero;

            float h = Radius;
            float factor = -45f / ((float)Math.PI * (float)Math.Pow(h, 6)) * (float)Math.Pow(h - rMagnitude, 2) / rMagnitude;

            return factor * r;
        }

        private float SpikyKernelSecondDerivative(float r)
        {
            float h = Radius;
            if (r >= h) return 0f;

            return 45f / ((float)Math.PI * (float)Math.Pow(h, 6)) * (h - r);
        }
    }

}
