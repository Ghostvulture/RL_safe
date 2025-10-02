#!/usr/bin/env python3

"""
Test script for the updated tolerance reward function.
"""

import numpy as np

# Mock implementation to test the tolerance function
class TestTolerance:
    def _sigmoids(self, x, value_at_1, sigmoid):
        """Various sigmoid functions for tolerance reward."""
        if sigmoid == 'gaussian':
            return value_at_1 * np.exp(-0.5 * x**2)
        elif sigmoid == 'linear':
            return np.where(x <= 1.0, (1.0 - x) + value_at_1 * x, 0.0)
        elif sigmoid == 'hyperbolic':
            return value_at_1 * (1.0 / (1.0 + x**2))
        elif sigmoid == 'long_tail':
            return value_at_1 * (2.0 / (1.0 + x**2))
        elif sigmoid == 'cosine':
            return value_at_1 * (0.5 * (1.0 + np.cos(np.pi * np.minimum(x, 1.0))))
        elif sigmoid == 'tanh_squared':
            return value_at_1 * (1.0 - np.tanh(x)**2)
        else:
            # Default to gaussian
            return value_at_1 * np.exp(-0.5 * x**2)

    def _tolerance_reward(self, x, bounds=(0.0, 0.0), margin=0.0, sigmoid='gaussian', value_at_margin=0.1):
        """Updated tolerance function matching dm_control API."""
        lower, upper = bounds
        if lower > upper:
            raise ValueError('Lower bound must be <= upper bound.')
        if margin < 0:
            raise ValueError('`margin` must be non-negative.')

        in_bounds = np.logical_and(lower <= x, x <= upper)
        if margin == 0:
            value = np.where(in_bounds, 1.0, 0.0)
        else:
            d = np.where(x < lower, lower - x, x - upper) / margin
            value = np.where(in_bounds, 1.0, self._sigmoids(d, value_at_margin, sigmoid))

        return float(value) if np.isscalar(x) else value

def test_tolerance_function():
    """Test the tolerance function with different parameters."""
    tester = TestTolerance()
    
    # Test parameters similar to the sparse reward
    target_vel = 0.5
    bounds = (target_vel, 2 * target_vel)  # (0.5, 1.0)
    margin = 2 * target_vel  # 1.0
    
    # Test specific values
    print("=== Tolerance Function Test Results ===")
    test_values = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5]
    
    print(f"Bounds: {bounds}")
    print(f"Margin: {margin}")
    print("Expected behavior:")
    print("- Velocity 0.5-1.0 m/s: reward = 1.0")
    print("- Velocity outside bounds but within margin: decreasing reward")
    print("- Velocity beyond margin: reward = 0.0")
    print()
    
    # Test different sigmoid types
    sigmoid_types = ['linear', 'gaussian', 'hyperbolic', 'cosine']
    
    for sigmoid in sigmoid_types:
        print(f"--- {sigmoid.upper()} SIGMOID ---")
        for vel in test_values:
            reward = tester._tolerance_reward(
                x=vel,
                bounds=bounds,
                margin=margin,
                value_at_margin=0.0,
                sigmoid=sigmoid
            )
            in_bounds = bounds[0] <= vel <= bounds[1]
            status = "IN BOUNDS" if in_bounds else "OUT OF BOUNDS"
            print(f"Velocity: {vel:4.1f} m/s -> Reward: {reward:.4f} ({status})")
        print()

if __name__ == "__main__":
    test_tolerance_function()

if __name__ == "__main__":
    test_tolerance_function()
