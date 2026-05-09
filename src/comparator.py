"""
Company Risk Comparator

Provides logic to load extracted risk profiles, aggregate data,
and generate comparison matrices between companies.
"""

import os
import json
import pandas as pd
from typing import List, Dict

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import COMPANIES, RISK_CATEGORIES

# Severity mapping for numeric scoring
SEVERITY_SCORE = {
    "high": 3,
    "medium": 2,
    "low": 1,
    "none": 0  # if is_present == False
}


class RiskComparator:
    """Loads risk profiles and provides data structures for comparison."""

    def __init__(self, profiles_dir: str = None):
        if profiles_dir is None:
            self.profiles_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "data", "risk_profiles"
            )
        else:
            self.profiles_dir = profiles_dir
            
        self.profiles = []
        self._load_profiles()

    def _load_profiles(self):
        """Load all available risk profiles."""
        if not os.path.exists(self.profiles_dir):
            return

        combined_path = os.path.join(self.profiles_dir, "all_risk_profiles.json")
        if os.path.exists(combined_path):
            with open(combined_path, "r", encoding="utf-8") as f:
                self.profiles = json.load(f)
        else:
            # Try to load individual files if combined doesn't exist
            for ticker in COMPANIES:
                path = os.path.join(self.profiles_dir, f"{ticker}_risk_profile.json")
                if os.path.exists(path):
                    with open(path, "r", encoding="utf-8") as f:
                        self.profiles.append(json.load(f))

    def get_available_companies(self) -> List[str]:
        """Return list of tickers with available profiles."""
        return [p["company"] for p in self.profiles]

    def get_company_profile(self, ticker: str) -> Dict | None:
        """Get the full profile for a specific company."""
        for p in self.profiles:
            if p["company"] == ticker:
                return p
        return None

    def get_risk_heatmap_data(self) -> pd.DataFrame:
        """
        Generate a matrix of companies vs risk categories with severity scores.
        Useful for building a heatmap.
        """
        data = []
        
        for profile in self.profiles:
            ticker = profile["company"]
            row = {"Company": ticker}
            
            for risk in profile["risk_assessments"]:
                cat = risk["risk_category"]
                if risk["is_present"]:
                    score = SEVERITY_SCORE.get(risk["severity"], 1)
                else:
                    score = 0
                row[cat] = score
                
            data.append(row)
            
        if not data:
            return pd.DataFrame()
            
        df = pd.DataFrame(data).set_index("Company")
        return df
    
    def get_severity_labels_matrix(self) -> pd.DataFrame:
        """Generate a matrix of text severity labels (Low, Medium, High)."""
        data = []
        
        for profile in self.profiles:
            ticker = profile["company"]
            row = {"Company": ticker}
            
            for risk in profile["risk_assessments"]:
                cat = risk["risk_category"]
                if risk["is_present"]:
                    row[cat] = risk["severity"].capitalize()
                else:
                    row[cat] = "None"
                
            data.append(row)
            
        if not data:
            return pd.DataFrame()
            
        df = pd.DataFrame(data).set_index("Company")
        return df

    def compare_two_companies(self, ticker1: str, ticker2: str) -> pd.DataFrame:
        """
        Compare two companies side-by-side across all risk categories.
        """
        p1 = self.get_company_profile(ticker1)
        p2 = self.get_company_profile(ticker2)
        
        if not p1 or not p2:
            raise ValueError("One or both company profiles not found.")
            
        data = []
        
        # Build dictionary for quick lookup
        r1 = {r["risk_category"]: r for r in p1["risk_assessments"]}
        r2 = {r["risk_category"]: r for r in p2["risk_assessments"]}
        
        for cat in RISK_CATEGORIES:
            cat_name = cat["name"]
            
            risk1 = r1.get(cat_name, {})
            risk2 = r2.get(cat_name, {})
            
            sev1 = risk1.get("severity", "none").capitalize() if risk1.get("is_present") else "None"
            sev2 = risk2.get("severity", "none").capitalize() if risk2.get("is_present") else "None"
            
            data.append({
                "Risk Category": cat_name,
                f"{ticker1} Severity": sev1,
                f"{ticker2} Severity": sev2,
                f"{ticker1} Explanation": risk1.get("explanation", ""),
                f"{ticker2} Explanation": risk2.get("explanation", "")
            })
            
        return pd.DataFrame(data)

    def get_top_risks_for_company(self, ticker: str, top_n: int = 3) -> List[Dict]:
        """Get the highest severity risks for a specific company."""
        profile = self.get_company_profile(ticker)
        if not profile:
            return []
            
        risks = [r for r in profile["risk_assessments"] if r["is_present"]]
        
        # Sort by severity score (High=3, Med=2, Low=1) then confidence
        risks.sort(key=lambda x: (SEVERITY_SCORE.get(x["severity"], 0), x["confidence"]), reverse=True)
        
        return risks[:top_n]


if __name__ == "__main__":
    # Quick test
    comp = RiskComparator()
    print("Available companies:", comp.get_available_companies())
