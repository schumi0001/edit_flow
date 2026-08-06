"""Legacy synthetic injection entry point.

Synthetic data injection has been removed from the live workflow. Use the live
Wikimedia producer instead to populate the Kafka input topic.
"""

print("Synthetic injection is disabled. Start the live Wikimedia producer to feed the pipeline.")
