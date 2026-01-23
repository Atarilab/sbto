def compute_success(df, 
                    max_pos=0.1, 
                    max_term_pos=0.1, 
                    max_quat=0.436332,
                    max_term_quat=0.436332,
                    ):
    """
    Returns a boolean Series marking which rows are successful.
    """
    return (
        (df['err_pos_obj'] <= max_pos) &
        (df['err_term_pos_obj'] <= max_term_pos) &
        (df['err_quat_obj'] <= max_quat) &
        (df['err_term_quat_obj'] <= max_term_quat)
    )
