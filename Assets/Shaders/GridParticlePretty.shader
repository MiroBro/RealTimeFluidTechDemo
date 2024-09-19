Shader "Instanced/FluidParticleShader" {
    Properties {
        _MainTex("Albedo (RGB)", 2D) = "white" {}
        _Glossiness("Smoothness", Range(0, 1)) = 0.8
        _Metallic("Metallic", Range(0, 1)) = 0.0
        _Color("Color", Color) = (0.25, 0.5, 0.5, 1)
        _DensityRange("Density Range", Range(0, 500000)) = 1.0
        _FresnelColor("Fresnel Color", Color) = (1, 1, 1, 1)
        _Refraction("Refraction", Range(0.0, 1.0)) = 0.02
        _BlendRadius("Blend Radius", Float) = 0.1
    }
    SubShader {
        Tags { "Queue" = "Transparent" "RenderType" = "Transparent" }
        LOD 200

        Blend SrcAlpha OneMinusSrcAlpha
        ZWrite Off

        CGPROGRAM
        #pragma surface surf Standard fullforwardshadows
        #pragma multi_compile_instancing
        #pragma instancing_options procedural:setup

        sampler2D _MainTex;
        float _Glossiness;
        float _Metallic;
        float4 _Color; // Ensure this is a float4 to include alpha
        float _DensityRange;
        float4 _FresnelColor; // Ensure this is a float4 to include alpha
        float _Refraction;
        float _BlendRadius;

        struct Input {
            float2 uv_MainTex;
            float3 viewDir;
            float3 worldPos; 
        };

        struct Particle {
            float pressure;
            float density;
            float3 currentForce;
            float3 velocity;
            float3 position;
        };

        #ifdef UNITY_PROCEDURAL_INSTANCING_ENABLED
        StructuredBuffer<Particle> _particlesBuffer;
        #endif

        void setup() {
        #ifdef UNITY_PROCEDURAL_INSTANCING_ENABLED
            float3 pos = _particlesBuffer[unity_InstanceID].position;
            float size = 1.0f; // Default size, adjust as necessary

            unity_ObjectToWorld._11_21_31_41 = float4(size, 0, 0, 0);
            unity_ObjectToWorld._12_22_32_42 = float4(0, size, 0, 0);
            unity_ObjectToWorld._13_23_33_43 = float4(0, 0, size, 0);
            unity_ObjectToWorld._14_24_34_44 = float4(pos.xyz, 1);
            unity_WorldToObject = unity_ObjectToWorld;
            unity_WorldToObject._14_24_34 *= -1;
            unity_WorldToObject._11_22_33 = 1.0f / unity_WorldToObject._11_22_33;
        #endif
        }

        void surf(Input IN, inout SurfaceOutputStandard o) {
            float4 texColor = tex2D(_MainTex, IN.uv_MainTex);

            #ifdef UNITY_PROCEDURAL_INSTANCING_ENABLED
            float dens = _particlesBuffer[unity_InstanceID].pressure;
            texColor.rgb = float3(dens / _DensityRange, 0, 0);
            #endif

            // Apply basic properties
            o.Albedo = texColor.rgb * _Color.rgb;
            o.Metallic = _Metallic;
            o.Smoothness = _Glossiness;
            o.Alpha = texColor.a * _Color.a;

            // Soft edge blending based on distance to nearby particles
            float distToEdge = length(IN.worldPos - _particlesBuffer[unity_InstanceID].position);
            float blendFactor = smoothstep(0.0, _BlendRadius, distToEdge);
            o.Alpha *= blendFactor;  // Use this factor to soften edges

            // Fresnel effect for the edges
            float fresnelEffect = pow(1.0f - dot(IN.viewDir, o.Normal), 3.0f);
            o.Emission = _FresnelColor.rgb * fresnelEffect;

            // Adjust Smoothness for a simple refraction effect
            o.Smoothness = lerp(o.Smoothness, 1.0, _Refraction);
        }
        ENDCG
    }
    FallBack "Diffuse"
}
