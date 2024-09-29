# RealTimeFluidTechDemo
Real Time Fluid Simulation using Smoothed Particle Hydrodynamics

This Repository contains a Unity project that compares the performance of the Smoothed Partcile Hydrodynamics (SPH).

**The specific Smoothed Partcile Hydrodynamics contained:**

0. GPU implementation of SPH
1. SPH C# code implemetation
2. SPH C# code optimized  with Burst 
3. SPH C# code optimized with Burst + additional optimization (removing sqrt operations, organizing liquid particles into chunks)
4. A separate, pure C#-project in .NET 8 CoreCLR written to compare Burst performance with default CoreCLR performance

**How to use:**
**Unity implementations:**
1. Pull down this project into a folder.
2. Download Unity 2022.3.37f
3. Open the project via Unity 2022.3.37f
4. Open the scene called "FluidSimulationScene"
5. In the scene in hierarchy, inside "Liquid Simulation" gameobject toggle on the type of simulation/type of code you want to inspect/profile (see screenshot below).
   
   ![image](https://github.com/user-attachments/assets/e92694b0-7875-4418-b386-6da8008b795e)
   
7. Run the scene/game. You can now inspect the performance in the Stats (FPS for example) and in the Profiler.

**.NET 8 / CoreCLR implementation:**
1. Using an IDE (like Visual Studio) Open the pure CoreCLR implementation of SPH - it's located in the project under CoreCLRComparison/FluidSimulationCoreCLR/FluidSimulationCoreCLR: https://github.com/MiroBro/RealTimeFluidTechDemo/tree/main/CoreCLRComparison/FluidSimulationCoreCLR/FluidSimulationCoreCLR
2. Run the program and make note of the microseconds the computations it takes per "tick" (added a screenshot below with example values)

![image](https://github.com/user-attachments/assets/a3da9952-b324-4732-b1df-020fcfed2dda)


3. One can also build the application with AOT settings (to squeaze out more performance) by opening cmd in CoreCLRComparison/FluidSimulationCoreCLR/FluidSimulationCoreCLR and running **dotnet publish -r win-x64 -c Release**. This will produce a runnable executable in RealTimeFluidTechDemo\CoreCLRComparison\FluidSimulationCoreCLR\FluidSimulationCoreCLR\bin\Release\net8.0\win-x64\publish

That's it! :)
