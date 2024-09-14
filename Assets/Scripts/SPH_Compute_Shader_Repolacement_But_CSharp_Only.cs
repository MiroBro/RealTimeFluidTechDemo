using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SPH_Compute_Shader_Repolacement_But_CSharp_Only : MonoBehaviour
{

    private struct Particle
    {
        public float pressure;
        public float density;
        public Vector3 currentForce;
        public Vector3 velocity;
        public Vector3 position;
    }

    private Particle[] _particles;

    // Variable Declarations 
    float particleMass = 2.0f;
    float viscosity = 200;
    float gasConstant = 2000;
    float restDensity = 300;
    float boundDamping = -0.5f;
    float radius = 2;
    float radius3 = 8;
    float radius2 = 4;
    float radius4 = 16;
    float radius5 = 32;
    float pi = 3.1415926535897932384626433832795028841971f;

    int particleLength;

    // Pre-computed
    float densityWeightConstant = 0.00497359197162172924277761760539f;
    float spikyGradient = -0.09947183943243458485555235210782f;
    float viscLaplacian = 0.39788735772973833942220940843129f;

    float timestep = 1;

    Vector3 boxSize;

    private float StdKernel(float distanceSquared)
    {
        // Doyub Kim
        float x = 1.0f - distanceSquared / radius2;
        return 315f / (64f * pi * radius3) * x * x * x;
    }

    public void ComputeDensityPressure (Vector3Int id)
    {
        Vector3 origin = _particles[id.x].position;
        float sum = 0;

        for (int i = 0; i < particleLength; i++)
        {
            Vector3 diff = origin - _particles[i].position;
            float distanceSquared = Vector3.Dot(diff,diff);

            if (radius2 * 0.004 > distanceSquared * 0.004)
            {
                float x = (radius2 * 0.004f ) - (distanceSquared * 0.004f );
                sum += StdKernel(distanceSquared * 0.004f);
            }
        }

        _particles[id.x].density = sum * particleMass * 0.000001f;
        _particles[id.x].pressure = gasConstant * (_particles[id.x].density - restDensity);

        if (_particles[id.x].pressure <= 0) _particles[id.x].pressure = 0;
    }

    // Doyub Kim page 130
    float SpikyKernelFirstDerivative(float distance)
    {
        float x = 1.0f - distance / radius;
        return -45.0f / (pi * radius4) * x * x;
    }

    // Doyub Kim page 130
    float SpikyKernelSecondDerivative(float distance)
    {
        // Btw, it derives 'distance' not 'radius' (h)
        float x = 1.0f - distance / radius;
        return 90f / (pi * radius5) * x;
    }

    Vector3 SpikyKernelGradient(float distance, Vector3 directionFromCenter)
    {
        return SpikyKernelFirstDerivative(distance) * directionFromCenter;
    }

    Vector3 spherePos;
    float sphereRadius;

    public void ComputeForces(Vector3Int id)
    {
        Vector3 origin = _particles[id.x].position;
        float density2 = _particles[id.x].density * _particles[id.x].density;
        Vector3 pressure = Vector3.zero;
        Vector3 visc = Vector3.zero;
        float mass2 = particleMass * particleMass;

        for (int i = 0; i < particleLength; i++)
        {

            if (origin.x == _particles[i].position.x && origin.y == _particles[i].position.y && origin.z == _particles[i].position.z)
            {
                continue;
            }

            float dist = Vector3.Distance(_particles[i].position, origin);
            if (dist < radius * 2)
            {
                Vector3 pressureGradientDirection = Vector3.Normalize(_particles[id.x].position - _particles[i].position);
                pressure += mass2 * (_particles[id.x].pressure / density2 + _particles[i].pressure / (_particles[i].density * _particles[i].density)) * SpikyKernelGradient(dist, pressureGradientDirection);   // Kim
                visc += viscosity * mass2 * (_particles[i].velocity - _particles[id.x].velocity) / _particles[i].density * SpikyKernelSecondDerivative(dist);
            }
        }

        _particles[id.x].currentForce = new Vector3(0, (-9.81f * particleMass), 0) - pressure + visc;

        // Handle Collision

        Vector3 colDir = _particles[id.x].position - spherePos;
        if (Vector3.Magnitude(colDir) < sphereRadius)
        {
            float mag = sphereRadius / Vector3.Magnitude(colDir);
            _particles[id.x].currentForce += colDir * 300 * mag;
        }

        // + pressure + visc/_particles[id.x].density;
    }

    public void Integrate(Vector3Int id)
    {
        // _particles[id.x].velocity += timestep * _particles[id.x].

        Vector3 topRight = boxSize / 2;
        Vector3 bottomLeft = -boxSize / 2;

        Vector3 vel = _particles[id.x].velocity + ((_particles[id.x].currentForce / particleMass) * timestep);
        _particles[id.x].position += vel * timestep;



        // Minimum Enforcements

        if (_particles[id.x].position.x - radius < bottomLeft.x)
        {
            vel.x *= boundDamping;
            _particles[id.x].position.x = bottomLeft.x + radius;
        }

        if (_particles[id.x].position.y - radius < bottomLeft.y)
        {
            vel.y *= boundDamping;
            _particles[id.x].position.y = bottomLeft.y + radius;
        }

        if (_particles[id.x].position.z - radius < bottomLeft.z)
        {
            vel.z *= boundDamping;
            _particles[id.x].position.z = bottomLeft.z + radius;
        }

        // Maximum Enforcements

        if (_particles[id.x].position.x + radius > topRight.x)
        {
            vel.x *= boundDamping;
            _particles[id.x].position.x = topRight.x - radius;
        }

        if (_particles[id.x].position.y + radius > topRight.y)
        {
            vel.y *= boundDamping;
            _particles[id.x].position.y = topRight.y - radius;
        }

        if (_particles[id.x].position.z + radius > topRight.z)
        {
            vel.z *= boundDamping;
            _particles[id.x].position.z = topRight.z - radius;
        }


        _particles[id.x].velocity = vel;
    }

}
