# Structural Anti-Patterns & Rewrite Examples

This reference documents structural "AI tells" (predictable layout, syntax, and paragraph patterns) alongside concrete Before/After rewrites.

---

## 1. Uniform Sentence Cadence (句式节奏僵化)

### Anti-Pattern
All sentences have nearly identical lengths and structure (Subject + Verb + Prepositional Modifier), creating a monotonic, robotic rhythm.

❌ **Before (AI Slop)**:
> High-performance computing requires careful memory optimization. Developers must manage cache line alignment to prevent false sharing. Profiling tools provide actionable insights into bandwidth bottlenecks. By analyzing hardware counters, teams can achieve significant throughput gains.

✅ **After (Anti-Slop)**:
> High-performance computing demands strict memory optimization. Without cache line alignment, false sharing degrades throughput fast. Use profiling tools to spot bandwidth bottlenecks, then inspect hardware counters directly to fix them.

---

## 2. Defensive Hedging & Filler Openings (垫字与防卫性客套)

### Anti-Pattern
Prefacing statements with unnecessary qualifiers like *"It is worth noting that"*, *"It is important to remember that"*, or *"不可否认的是"*.

❌ **Before (AI Slop)**:
> It is important to note that while quantizing LLMs reduces memory consumption, it is worth remembering that precision loss may impact accuracy in edge cases.

✅ **After (Anti-Slop)**:
> Quantizing LLMs reduces memory usage, but precision loss can degrade accuracy in edge cases.

---

## 3. Forced Symmetrical Summaries (强行结尾展望与升华)

### Anti-Pattern
Ending every section or response with a generic, overly optimistic wrap-up sentence.

❌ **Before (AI Slop)**:
> We replaced the global lock with fine-grained mutexes in the connection pool. Ultimately, this change stands as a testament to our team's commitment to reliability, paving the way for a more robust infrastructure.

✅ **After (Anti-Slop)**:
> We replaced the global lock with fine-grained mutexes in the connection pool. Benchmark results show contention dropped by 74%.

---

## 4. Forced Bullet List Abuse (滥用粗体标题列表)

### Anti-Pattern
Converting clear narrative prose into repetitive bullet points where every item starts with a bolded 2-3 word title.

❌ **Before (AI Slop)**:
To optimize your workflow:
- **Analyze Codebase**: Review all current dependencies to find conflicts.
- **Implement Caching**: Add Redis caching to reduce database read latencies.
- **Monitor Metrics**: Set up Prometheus alerts to track latency spikes.

✅ **After (Anti-Slop)**:
Start by reviewing dependencies for conflicts. Next, add Redis caching to reduce database reads and set up Prometheus to alert on latency spikes.

---

## 5. Overuse of Em-Dashes (滥用破折号)

### Anti-Pattern
Using em-dashes (`—`) in almost every paragraph to inject commentary or force appositives.

❌ **Before (AI Slop)**:
> The new serving engine—built from the ground up with custom C++ kernels—achieved higher throughput—far exceeding our initial targets—while maintaining sub-10ms latency.

✅ **After (Anti-Slop)**:
> Built with custom C++ kernels, the new serving engine achieved higher throughput than targeted while maintaining sub-10ms latency.

---

## 6. Fake Opposition ("不是X，而是Y")

### Anti-Pattern
Asserting a contrast the sentence never proves. The "不是" half is usually already implied by the "而是" half. Skeleton test: delete the "不是X" half and read what remains.

❌ **Before (AI Slop)**:
> 量化不是一个简单的精度问题，而是一个系统级的效率问题。

✅ **After (Anti-Slop)**:
> 量化影响整条链路：权重、激活、kernel 选择互相牵制，只调精度不够。

❌ **Before (AI Slop)**:
> It's not about writing faster code, it's about writing less code.

✅ **After (Anti-Slop)**:
> Write less code. Fewer instructions mean fewer cache misses.

---

## 7. Assert-Then-Retract ("X，但这不代表Y")

### Anti-Pattern
Making a claim and retracting it in the same breath, hedging against a criticism no one made. The sentence is two positions glued together; the author takes neither.

❌ **Before (AI Slop)**:
> 扩展集群能提升吞吐，但这并不意味着网络不再是瓶颈。

✅ **After (Anti-Slop)**:
> 网络仍是瓶颈。加 10 台机器，吞吐只涨了 15%。

---

## 8. Synonym Cycling (同义并列)

### Anti-Pattern
Three parallel items with one meaning. Parallel structure padded with repetition. Test each item: does it differ from the previous one?

❌ **Before (AI Slop)**:
> 我们评估了系统的可扩展性、可伸缩性与可延展性。

✅ **After (Anti-Slop)**:
> 我们测试了从 10 台扩展到 1000 台的吞吐曲线。

❌ **Before (AI Slop)**:
> The engine must be fast, efficient, and performant.

✅ **After (Anti-Slop)**:
> The engine must hold 3ms p99 at 200 QPS.
