for tau in 0.1; do
    for prediction_method in ilstm; do
        for mutable in --mutable; do
            echo tau = $tau
            echo prediction_method = $prediction_method
            echo mutable = $mutable
            python -u scripts/offline_filtering_process_hri.py --tau $tau --prediction_method $prediction_method \
                $mutable | tee -a logs/211109.txt
        done
    done
done