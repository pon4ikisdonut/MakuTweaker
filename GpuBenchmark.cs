using System;
using System.Diagnostics;
using System.Numerics;
using System.Threading;
using SharpGen.Runtime;
using Vortice.D3DCompiler;
using Vortice.Direct3D;
using Vortice.Direct3D11;
using Vortice.DXGI;

namespace MakuTweakerNew
{
    public class GpuBenchmarkResult
    {
        public double GFlops { get; set; }
        public double MemoryBandwidthGBps { get; set; }
        public double FinalScore { get; set; }
        public bool Success { get; set; }
    }

    public static class GpuBenchmark
    {

        // fp32 fma
        private const string ComputeShaderCode = @"
            RWStructuredBuffer<float4> OutputBuffer : register(u0);

            cbuffer Params : register(b0)
            {
                float4 kParam;
            };

            [numthreads(256, 1, 1)]
            void CSCompute(uint3 id : SV_DispatchThreadID)
            {
                float4 a = float4(id.x * 0.0001f, 0.1f, 0.2f, 0.3f);
                float4 b = float4(0.2f, id.x * 0.0001f, 0.4f, 0.5f);
                float4 c = float4(0.3f, 0.4f, id.x * 0.0001f, 0.6f);
                float4 d = float4(0.4f, 0.5f, 0.6f, id.x * 0.0001f);
                float4 k = kParam;

                [unroll]
                for (int i = 0; i < 64; i++)
                {
                    // 4 независимые цепочки mad без задержек регистров
                    a = mad(a, k, 0.1f);
                    b = mad(b, k, 0.2f);
                    c = mad(c, k, 0.3f);
                    d = mad(d, k, 0.4f);
                }

                OutputBuffer[id.x] = a + b + c + d;
            }
        ";

        // потоковое параллельное чтение 4 независимых стримов
        private const string MemoryShaderCode = @"
            StructuredBuffer<float4> BufferA : register(t0);
            RWStructuredBuffer<float4> OutputBuffer : register(u0);

            [numthreads(256, 1, 1)]
            void CSMemory(uint3 id : SV_DispatchThreadID)
            {
                uint groupBase = (id.x / 256) * 16384;
                uint threadInGroup = id.x % 256;
                uint baseIdx = groupBase + threadInGroup;

                float4 s0 = 0;
                float4 s1 = 0;
                float4 s2 = 0;
                float4 s3 = 0;

                [unroll]
                for (int i = 0; i < 16; i++)
                {
                    // 4 параллельных независимых запроса в разные каналы памяти
                    s0 += BufferA[baseIdx + 256 * i];
                    s1 += BufferA[baseIdx + 256 * (i + 16)];
                    s2 += BufferA[baseIdx + 256 * (i + 32)];
                    s3 += BufferA[baseIdx + 256 * (i + 48)];
                }

                OutputBuffer[id.x] = s0 + s1 + s2 + s3;
            }
        ";

        // ждем данные аппаратного счетчика с защитой от вечного зависания и tdr
        private static bool WaitForQueryData<T>(ID3D11Device device, ID3D11DeviceContext context, ID3D11Query query, out T data, int timeoutMs = 2000) where T : unmanaged
        {
            var sw = Stopwatch.StartNew();
            while (!context.GetData<T>(query, out data))
            {
                if (device.DeviceRemovedReason.Failure || sw.ElapsedMilliseconds > timeoutMs)
                {
                    data = default;
                    return false;
                }
                Thread.Yield();
            }
            return true;
        }

        // запуск полного цикла тестов карты
        public static GpuBenchmarkResult Run(string? targetGpuName = null, ulong vramBytes = 0)
        {
            var result = new GpuBenchmarkResult();
            ID3D11Device? device = null;
            ID3D11DeviceContext? context = null;
            IDXGIAdapter1? selectedAdapter = null;

            try
            {
                // ищем нужный адаптер через dxgi
                using var factory = DXGI.CreateDXGIFactory1<IDXGIFactory1>();
                if (factory != null)
                {
                    int i = 0;
                    IDXGIAdapter1? bestAdapter = null;
                    ulong maxVram = 0;

                    while (factory.EnumAdapters1((uint)i, out var adapter).Success && adapter != null)
                    {
                        var desc = adapter.Description1;
                        bool isSoftware = (desc.Flags & AdapterFlags.Software) != 0;
                        if (!isSoftware)
                        {
                            string descName = desc.Description?.Trim() ?? "";
                            if (!string.IsNullOrEmpty(targetGpuName) &&
                                (descName.Equals(targetGpuName, StringComparison.OrdinalIgnoreCase) ||
                                 descName.Contains(targetGpuName, StringComparison.OrdinalIgnoreCase) ||
                                 targetGpuName.Contains(descName, StringComparison.OrdinalIgnoreCase)))
                            {
                                selectedAdapter?.Dispose();
                                selectedAdapter = adapter;
                                bestAdapter?.Dispose();
                                bestAdapter = null;
                                break;
                            }

                            if (desc.DedicatedVideoMemory > maxVram || bestAdapter == null)
                            {
                                maxVram = desc.DedicatedVideoMemory;
                                bestAdapter?.Dispose();
                                bestAdapter = adapter;
                            }
                            else
                            {
                                adapter.Dispose();
                            }
                        }
                        else
                        {
                            adapter.Dispose();
                        }
                        i++;
                    }

                    if (selectedAdapter == null)
                        selectedAdapter = bestAdapter;
                }

                // d3d11 хэдлэсс создание устройства 
                FeatureLevel[] levels = { FeatureLevel.Level_11_1, FeatureLevel.Level_11_0, FeatureLevel.Level_10_1, FeatureLevel.Level_10_0 };
                var hr = D3D11.D3D11CreateDevice(
                    selectedAdapter,
                    selectedAdapter != null ? DriverType.Unknown : DriverType.Hardware,
                    DeviceCreationFlags.None,
                    levels,
                    out device,
                    out context);

                if (vramBytes == 0 && selectedAdapter != null)
                {
                    vramBytes = selectedAdapter.Description1.DedicatedVideoMemory;
                }

                if (hr.Failure || device == null || context == null)
                {
                    selectedAdapter?.Dispose();
                    return FallbackEstimate(vramBytes);
                }

                // запуск замера пиковых терафлопсов (1.5сек)
                double gflops = MeasureCompute(device, context);

                // замер скорости памяти (1.5сек)
                double memGBps = MeasureMemory(device, context);

                // считаем баллы от 0 до 10 по честным замерам
                double vramGb = vramBytes / (1024.0 * 1024 * 1024);
                double scoreCompute = Math.Clamp(1.15 * Math.Log(1.0 + 1.5 * (gflops / 1000.0)), 0.2, 4.5);
                double scoreMem = Math.Clamp(0.85 * Math.Log(1.0 + 0.05 * memGBps), 0.1, 3.5);
                double scoreVram = Math.Clamp(vramGb * (2.0 / 16.0), 0.2, 2.0);

                result.GFlops = gflops;
                result.MemoryBandwidthGBps = memGBps;
                result.FinalScore = Math.Clamp(scoreCompute + scoreMem + scoreVram, 0.5, 10.0);
                result.Success = true;
                return result;
            }
            catch
            {
                // если совсем все сдохло наху то отдаем фоллбэк
                return FallbackEstimate(vramBytes);
            }
            finally
            {
                // убираем за собой то что насрали в процессе теста
                context?.Dispose();
                device?.Dispose();
                selectedAdapter?.Dispose();
            }
        }

        // замер мощности фма (~1500 мс)
        private static double MeasureCompute(ID3D11Device device, ID3D11DeviceContext context)
        {
            ReadOnlyMemory<byte> shaderBytecode = Compiler.Compile(ComputeShaderCode, "CSCompute", "CSCompute", "cs_5_0");
            using var computeShader = device.CreateComputeShader(shaderBytecode.Span);

            // 1 млн float4 элементов (4096 групп потоков) для полной загрузки всех ядер
            const int elementCount = 1048576;
            const int bufferSize = elementCount * 16;

            var bufferDesc = new BufferDescription
            {
                ByteWidth = (uint)bufferSize,
                Usage = ResourceUsage.Default,
                BindFlags = BindFlags.UnorderedAccess | BindFlags.ShaderResource,
                StructureByteStride = 16,
                MiscFlags = ResourceOptionFlags.BufferStructured
            };

            var uavDesc = new UnorderedAccessViewDescription
            {
                Format = Vortice.DXGI.Format.Unknown,
                ViewDimension = UnorderedAccessViewDimension.Buffer,
                Buffer = new BufferUnorderedAccessView { FirstElement = 0, NumElements = (uint)elementCount }
            };

            // кольцо уав для вычислений чтоб убрать сериализацию конвейера
            const int uavRingSize = 4;
            var computeBuffers = new ID3D11Buffer[uavRingSize];
            var computeUavs = new ID3D11UnorderedAccessView[uavRingSize];
            for (int r = 0; r < uavRingSize; r++)
            {
                computeBuffers[r] = device.CreateBuffer(bufferDesc);
                computeUavs[r] = device.CreateUnorderedAccessView(computeBuffers[r], uavDesc);
            }

            // константный буфер с множителем чтоб компилятор не вырезал математику
            var cbDesc = new BufferDescription
            {
                ByteWidth = 16,
                Usage = ResourceUsage.Default,
                BindFlags = BindFlags.ConstantBuffer
            };
            using var cb = device.CreateBuffer(new Vector4(0.9999f, 0.9999f, 0.9999f, 0.9999f), cbDesc);

            // аппаратные таймштампы для чистого времени исполнения на кристалле
            using var disjointQuery = device.CreateQuery(new QueryDescription { QueryType = QueryType.TimestampDisjoint });
            using var startQuery = device.CreateQuery(new QueryDescription { QueryType = QueryType.Timestamp });
            using var endQuery = device.CreateQuery(new QueryDescription { QueryType = QueryType.Timestamp });

            context.CSSetShader(computeShader);
            context.CSSetConstantBuffer(0, cb);
            context.CSSetUnorderedAccessView(0, computeUavs[0]);

            // запуск прогревочного прогона
            context.Dispatch((uint)(elementCount / 256), 1, 1);
            context.Flush();

            // запуск пачками по 16 штук с замером времени работы чипа 
            const int batchSize = 16;
            int completedDispatches = 0;
            double totalGpuSeconds = 0;
            var sw = Stopwatch.StartNew();

            while (sw.ElapsedMilliseconds < 1500)
            {
                context.Begin(disjointQuery);
                context.End(startQuery);
                for (int b = 0; b < batchSize; b++)
                {
                    // крутим уав по очереди чтоб не ловить блокировки
                    context.CSSetUnorderedAccessView(0, computeUavs[b % uavRingSize]);
                    context.Dispatch((uint)(elementCount / 256), 1, 1);
                }
                context.End(endQuery);
                context.End(disjointQuery);
                context.Flush();

                // забираем таймштампы без риска вечного зависания при сбое драйвера
                if (!WaitForQueryData(device, context, disjointQuery, out QueryDataTimestampDisjoint disjointData) ||
                    !WaitForQueryData(device, context, startQuery, out ulong startTime) ||
                    !WaitForQueryData(device, context, endQuery, out ulong endTime))
                {
                    // вылет по tdr или сброс драйвера
                    break;
                }

                if (!disjointData.Disjoint && disjointData.Frequency > 0 && endTime > startTime)
                {
                    totalGpuSeconds += (double)(endTime - startTime) / disjointData.Frequency;
                    completedDispatches += batchSize;
                }
            }
            sw.Stop();

            // чистим уав и константник чтоб не мусорить
            context.CSSetShader(null);
            context.CSSetConstantBuffer(0, null);
            context.CSSetUnorderedAccessView(0, null);

            // прибираем буферы вычислений
            for (int r = 0; r < uavRingSize; r++)
            {
                computeUavs[r]?.Dispose();
                computeBuffers[r]?.Dispose();
            }

            // считаем реальные терафлопсы: 64 итерации * 4 mad * 2 флопа * 4 компонента = 2048 флоп на поток
            const double flopsPerThread = 64.0 * 4.0 * 2.0 * 4.0;
            double totalFlops = (double)elementCount * flopsPerThread * completedDispatches;
            double measureSeconds = totalGpuSeconds > 0 ? totalGpuSeconds : sw.Elapsed.TotalSeconds;
            double gflops = (totalFlops / measureSeconds) / 1_000_000_000.0;
            return gflops;
        }

        // замер скорости шины видеопамяти на ~1500 мс
        private static double MeasureMemory(ID3D11Device device, ID3D11DeviceContext context)
        {
            // 262144 потока * 64 float4 = 16 млн float4 = 256 мб буфер (точно не попадет в l2 кэш карты)
            int threads = 262144;
            int totalElements = 16777216;
            ID3D11Buffer? bufferA = null;

            try
            {
                var bufferDesc = new BufferDescription
                {
                    ByteWidth = (uint)(totalElements * 16),
                    Usage = ResourceUsage.Default,
                    BindFlags = BindFlags.ShaderResource,
                    StructureByteStride = 16,
                    MiscFlags = ResourceOptionFlags.BufferStructured
                };
                bufferA = device.CreateBuffer(bufferDesc);
            }
            catch
            {
                // безопасный откат на 128 мб только если видеопамяти критически мало
                threads = 131072;
                totalElements = 8388608;
                var bufferDesc = new BufferDescription
                {
                    ByteWidth = (uint)(totalElements * 16),
                    Usage = ResourceUsage.Default,
                    BindFlags = BindFlags.ShaderResource,
                    StructureByteStride = 16,
                    MiscFlags = ResourceOptionFlags.BufferStructured
                };
                bufferA?.Dispose();
                bufferA = device.CreateBuffer(bufferDesc);
            }

            using var inBuf = bufferA;

            // компилим шейдер памяти с 4 параллельными выборками
            ReadOnlyMemory<byte> shaderBytecode = Compiler.Compile(MemoryShaderCode, "CSMemory", "CSMemory", "cs_5_0");
            using var memoryShader = device.CreateComputeShader(shaderBytecode.Span);

            using var srv = device.CreateShaderResourceView(inBuf);

            // выходные буферы записи по 4 мб для исключения uav hazard
            var outDesc = new BufferDescription
            {
                ByteWidth = (uint)(threads * 16),
                Usage = ResourceUsage.Default,
                BindFlags = BindFlags.UnorderedAccess,
                StructureByteStride = 16,
                MiscFlags = ResourceOptionFlags.BufferStructured
            };

            var uavDesc = new UnorderedAccessViewDescription
            {
                Format = Vortice.DXGI.Format.Unknown,
                ViewDimension = UnorderedAccessViewDimension.Buffer,
                Buffer = new BufferUnorderedAccessView { FirstElement = 0, NumElements = (uint)threads }
            };

            const int uavRingSize = 4;
            var outBuffers = new ID3D11Buffer[uavRingSize];
            var outUavs = new ID3D11UnorderedAccessView[uavRingSize];
            for (int r = 0; r < uavRingSize; r++)
            {
                outBuffers[r] = device.CreateBuffer(outDesc);
                outUavs[r] = device.CreateUnorderedAccessView(outBuffers[r], uavDesc);
            }

            // аппаратные таймштампы для замера чистого времени чтения на чипе
            using var disjointQuery = device.CreateQuery(new QueryDescription { QueryType = QueryType.TimestampDisjoint });
            using var startQuery = device.CreateQuery(new QueryDescription { QueryType = QueryType.Timestamp });
            using var endQuery = device.CreateQuery(new QueryDescription { QueryType = QueryType.Timestamp });

            context.CSSetShader(memoryShader);
            context.CSSetShaderResource(0, srv);
            context.CSSetUnorderedAccessView(0, outUavs[0]);

            // запуск прогревочного прогона памяти
            int groupCount = threads / 256;
            context.Dispatch((uint)groupCount, 1, 1);
            context.Flush();

            // запуск теста памяти пачками по 8 штук 
            const int batchSize = 8;
            int completedDispatches = 0;
            double totalGpuSeconds = 0;
            var sw = Stopwatch.StartNew();

            while (sw.ElapsedMilliseconds < 1500)
            {
                context.Begin(disjointQuery);
                context.End(startQuery);
                for (int b = 0; b < batchSize; b++)
                {
                    // крутим уав по очереди чтобы исключить uav hazard serialization
                    context.CSSetUnorderedAccessView(0, outUavs[b % uavRingSize]);
                    context.Dispatch((uint)groupCount, 1, 1);
                }
                context.End(endQuery);
                context.End(disjointQuery);
                context.Flush();

                if (!WaitForQueryData(device, context, disjointQuery, out QueryDataTimestampDisjoint disjointData) ||
                    !WaitForQueryData(device, context, startQuery, out ulong startTime) ||
                    !WaitForQueryData(device, context, endQuery, out ulong endTime))
                {
                    // вылет по tdr или сброс драйвера
                    break;
                }

                if (!disjointData.Disjoint && disjointData.Frequency > 0 && endTime > startTime)
                {
                    totalGpuSeconds += (double)(endTime - startTime) / disjointData.Frequency;
                    completedDispatches += batchSize;
                }
            }
            sw.Stop();

            // чистим биндинги без лишних аллокаций
            context.CSSetShader(null);
            context.CSSetShaderResource(0, null);
            context.CSSetUnorderedAccessView(0, null);

            // прибираем кольцо буферов памяти
            for (int r = 0; r < uavRingSize; r++)
            {
                outUavs[r]?.Dispose();
                outBuffers[r]?.Dispose();
            }

            // честный расчет скорости: переданные байты делим на чистое время работы чипа
            double bytesPerDispatch = (double)totalElements * 16.0 + (double)threads * 16.0;
            double totalBytes = bytesPerDispatch * completedDispatches;
            double measureSeconds = totalGpuSeconds > 0 ? totalGpuSeconds : sw.Elapsed.TotalSeconds;
            double gbps = (totalBytes / measureSeconds) / 1_000_000_000.0;
            return gbps;
        }

        // фоллбэк если девайс совсем не поддерживает compute shader 
        // (хотя это маловероятно на современных картах, только если это не всякие nvidia gtx 260/280, gts 250, ati radeon hd 2000-4000, 
        // hd graphics 2/3000 или чет такое, короче т о что не поддерживает d3d11)
        private static GpuBenchmarkResult FallbackEstimate(ulong vramBytes = 0)
        {
            double vramGb = vramBytes / (1024.0 * 1024 * 1024);
            return new GpuBenchmarkResult
            {
                GFlops = 0,
                MemoryBandwidthGBps = 0,
                FinalScore = Math.Clamp(vramGb * (10.0 / 16.0), 0.0, 10.0),
                Success = false
            };
        }
    }
}
