from datetime import datetime

def transform_date_format(date_str):
    """
    Convert various date string formats to the format "%d-%m-%Y".
    
    Parameters:
        date_str (str): Date string in one of the recognized formats.
        
    Returns:
        str: Date string in the format "%d-%m-%Y".
    """
    formats = ["%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y"]
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt).strftime("%d-%m-%Y")
        except ValueError:
            continue
    raise ValueError(f"Unknown date format: {date_str}")

def transform_plan_name(plan_name):
    """
    Simplify plan names.
    
    Parameters:
        plan_name (str): The plan name to be transformed.
        
    Returns:
        str: Transformed plan name.
    """
    mapping = {
        'GoldPlan': 'Gold',
        'SilverPlan': 'Silver',
        'BronzePlan': 'Bronze',
        'GoldPackage': 'Gold',
        'SilverPackage': 'Silver',
        'BronzePackage': 'Bronze'
    }
    return mapping.get(plan_name, plan_name)

