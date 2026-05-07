def metric_card(label, value):
    """Statistic panel HTML string."""
    return f"""
    <div class='stat-card'>
        <div class='label'>{label}</div>
        <div class='value'>{value}</div>
    </div>
    """

def info_panel(text):
    """Info panel HTML string."""
    return f"<div class='info-panel'>{text}</div>"

def warn_panel(text):
    """Warning panel HTML string."""
    return f"<div class='warn-panel'>{text}</div>"

def success_panel(text):
    """Success panel HTML string."""
    return f"<div class='success-panel'>{text}</div>"

def section_label(label):
    """Section label panel HTML string."""
    return f"<div class='section-label'>{label}</div>"

