import math
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from peak_height import specific_heat, fit_peak, fit_log

class PeakHeightTests(unittest.TestCase):
    def test_centered_curvature(self):
        k=np.linspace(.4,.5,11); x,y=specific_heat(k,k*k)
        np.testing.assert_allclose(y,2*x*x,rtol=1e-11)
    def test_quadratic_peak(self):
        k=np.linspace(.42,.46,9); y=3-80*(k-.441)**2
        x,height=fit_peak(k,y)
        self.assertAlmostEqual(x,.441,places=10);self.assertAlmostEqual(height,3,places=10)
    def test_log_fit(self):
        sizes=np.array([64,256,1024]);y=1+2*np.log(sizes)
        slope,intercept=fit_log(sizes,y)
        self.assertAlmostEqual(slope,2,places=10);self.assertAlmostEqual(intercept,1,places=10)

class RefinedGridTests(unittest.TestCase):
    def test_half_microkelvin_spacing_is_preserved(self):
        from peak_height_fine import fine_grid
        grid=fine_grid(.44057686523111605,440000)
        self.assertEqual(len(grid),9)
        self.assertTrue(all(b-a==312500 for a,b in zip(grid,grid[1:])))
        self.assertEqual(grid[4],440625000)

if __name__=='__main__': unittest.main()
