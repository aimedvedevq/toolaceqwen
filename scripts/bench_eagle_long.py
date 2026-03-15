#!/usr/bin/env python3
"""Benchmark EAGLE-3 on long multi-turn tool-calling conversations."""

import json
import time
import numpy as np
import sglang

# Multi-turn tool calling scenarios — model generates long responses
SCENARIOS = [
    # 1. Multi-step research task
    {
        "messages": [
            {"role": "system", "content": "You are a research assistant with access to: search(query), get_page(url), summarize(text), translate(text, lang), analyze_sentiment(text), extract_entities(text), compare(text1, text2), generate_report(title, sections). Use tools step by step. Explain your reasoning before each tool call."},
            {"role": "user", "content": "I need a comprehensive analysis of recent AI developments. Search for the latest news, get details from top 3 results, summarize each, extract key entities, analyze sentiment, and generate a final report comparing the findings."},
        ],
        "max_tokens": 1024,
    },
    # 2. Multi-step data analysis
    {
        "messages": [
            {"role": "system", "content": "You are a data analyst with tools: query_database(sql), create_chart(data, chart_type), calculate_statistics(data), filter_data(data, conditions), join_datasets(ds1, ds2, key), export_csv(data, filename), send_email(to, subject, body, attachments). Think step by step."},
            {"role": "user", "content": "Analyze our Q4 sales data. Query the sales database for Q4 2025, calculate statistics (mean, median, std dev) by region, create bar and line charts, filter for top performing products, join with customer demographics, export results to CSV, and email the report to the team with all charts attached."},
        ],
        "max_tokens": 1024,
    },
    # 3. Complex travel planning
    {
        "messages": [
            {"role": "system", "content": "You are a travel agent with tools: search_flights(from, to, date, passengers), search_hotels(city, checkin, checkout, guests, stars), get_weather(city, date), convert_currency(amount, from_currency, to_currency), book_restaurant(city, cuisine, date, guests), get_attractions(city), calculate_budget(items), create_itinerary(days). Explain each step."},
            {"role": "user", "content": "Plan a 7-day trip to Japan for a family of 4 starting April 1st. We need flights from San Francisco, hotels in Tokyo (3 nights) and Kyoto (4 nights), restaurant reservations for each evening (mix of sushi, ramen, tempura, and kaiseki), weather forecasts for both cities, top attractions in each city, currency conversion for a $5000 budget, and a detailed day-by-day itinerary."},
        ],
        "max_tokens": 1024,
    },
    # 4. Code review and debugging
    {
        "messages": [
            {"role": "system", "content": "You are a senior developer with tools: read_file(path), search_code(pattern, directory), run_tests(test_path), lint_code(file), check_dependencies(package), get_git_log(n), create_pull_request(title, description, files), deploy(environment). Explain your analysis in detail."},
            {"role": "user", "content": "Review the authentication module. Read the main auth file, search for any SQL injection vulnerabilities, check for hardcoded credentials, run the test suite, lint the code for style issues, check if all dependencies are up to date, review the last 10 git commits for suspicious changes, and create a detailed PR with your findings and fixes."},
        ],
        "max_tokens": 1024,
    },
    # 5. Customer support escalation
    {
        "messages": [
            {"role": "system", "content": "You are a customer support agent with tools: lookup_customer(id), get_order_history(customer_id), check_inventory(product_id), process_refund(order_id, amount, reason), create_ticket(priority, description), send_notification(customer_id, message), update_order_status(order_id, status), get_shipping_info(tracking_number), apply_discount(customer_id, percentage, reason). Be thorough and empathetic."},
            {"role": "user", "content": "Customer #12345 is very upset. Their order #98765 arrived damaged, they also received the wrong color for order #98766, and they're asking about order #98767 which hasn't shipped yet. They want refunds for the first two orders, expedited shipping for the third, and a loyalty discount for future orders. Please handle everything, check all order details, inventory for replacements, process refunds, update statuses, send appropriate notifications, and create a high-priority ticket summarizing everything."},
        ],
        "max_tokens": 1024,
    },
] * 4  # 20 scenarios total


def run_bench(engine, label, scenarios, max_tokens=1024):
    """Run benchmark on long scenarios."""
    latencies = []
    tokens_list = []

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("./output_grpo/merged", trust_remote_code=True)

    for i, scenario in enumerate(scenarios):
        prompt = tok.apply_chat_template(
            scenario["messages"], tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )

        t0 = time.perf_counter()
        result = engine.generate(
            prompt=prompt,
            sampling_params={"max_new_tokens": scenario.get("max_tokens", max_tokens), "temperature": 0},
        )
        t1 = time.perf_counter()
        lat = (t1 - t0) * 1000
        ntok = result["meta_info"]["completion_tokens"]
        latencies.append(lat)
        tokens_list.append(ntok)

        if i < 3:
            text = result["text"][:150].replace("\n", " ")
            print(f"  [{i}] {ntok} tok, {lat:.0f}ms ({ntok/(lat/1000):.0f} tok/s): {text}...")

    latencies = np.array(latencies)
    tokens_arr = np.array(tokens_list)
    total_tokens = int(tokens_arr.sum())
    total_time = latencies.sum() / 1000

    stats = {
        "label": label,
        "n": len(scenarios),
        "avg_tokens": float(tokens_arr.mean()),
        "total_tokens": total_tokens,
        "latency_p50": float(np.percentile(latencies, 50)),
        "latency_p95": float(np.percentile(latencies, 95)),
        "tps": float(total_tokens / total_time) if total_time > 0 else 0,
        "ms_per_token": float(total_time / total_tokens * 1000) if total_tokens > 0 else 0,
    }

    print(f"\n  {label}: avg_tok={stats['avg_tokens']:.0f}, lat_p50={stats['latency_p50']:.0f}ms, "
          f"tps={stats['tps']:.0f}, ms/tok={stats['ms_per_token']:.2f}")
    return stats


def main():
    results = {}

    # Baseline
    print("=" * 60)
    print("BASELINE")
    print("=" * 60)
    engine = sglang.Engine(model_path="./output_grpo/merged", mem_fraction_static=0.7)
    results["baseline"] = run_bench(engine, "Baseline", SCENARIOS)
    engine.shutdown()
    time.sleep(5)

    # Finetuned EAGLE3
    print("\n" + "=" * 60)
    print("FINETUNED EAGLE-3")
    print("=" * 60)
    engine = sglang.Engine(
        model_path="./output_grpo/merged", mem_fraction_static=0.7,
        speculative_algorithm="EAGLE3",
        speculative_draft_model_path="./output_eagle/finetuned_sglang",
        speculative_num_steps=5,
        speculative_eagle_topk=4,
        speculative_num_draft_tokens=16,
    )
    results["eagle3_ft"] = run_bench(engine, "EAGLE-3 Finetuned", SCENARIOS)
    engine.shutdown()

    # Summary
    b = results["baseline"]
    e = results["eagle3_ft"]
    speedup = e["tps"] / b["tps"] if b["tps"] > 0 else 0
    lat_speedup = b["latency_p50"] / e["latency_p50"] if e["latency_p50"] > 0 else 0

    print(f"\n{'='*70}")
    print(f"{'':25s} {'Baseline':>15s} {'EAGLE-3 FT':>15s} {'Speedup':>10s}")
    print(f"{'-'*70}")
    print(f"  {'Avg tokens':25s} {b['avg_tokens']:>14.0f} {e['avg_tokens']:>14.0f}")
    print(f"  {'Latency p50':25s} {b['latency_p50']:>13.0f}ms {e['latency_p50']:>13.0f}ms {lat_speedup:>9.2f}x")
    print(f"  {'Latency p95':25s} {b['latency_p95']:>13.0f}ms {e['latency_p95']:>13.0f}ms")
    print(f"  {'Sequential TPS':25s} {b['tps']:>14.0f} {e['tps']:>14.0f} {speedup:>9.2f}x")
    print(f"  {'ms/token':25s} {b['ms_per_token']:>13.2f}ms {e['ms_per_token']:>13.2f}ms")
    print(f"{'='*70}")

    with open("bench_eagle_long.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("Saved to bench_eagle_long.json")


if __name__ == "__main__":
    main()
