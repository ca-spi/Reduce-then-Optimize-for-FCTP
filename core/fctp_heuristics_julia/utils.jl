module UtilityFunctions

export evaluate_sol
export sol_dict_to_matrix
export node_cycle_to_edge_loop
export reorder_loop
export write_to_log

#############################################
# General helper functions
#############################################

function evaluate_sol(sol::AbstractMatrix{Int}, var_costs::AbstractMatrix{Float64}, fix_costs::AbstractMatrix{Float64})
    total_var_costs = sol .* var_costs
    total_fix_costs = (sol .> 0) .* fix_costs
    return sum(total_var_costs .+ total_fix_costs)
end

function evaluate_sol(sol::Dict{Tuple{Int,Int},Int}, var_costs::AbstractMatrix{Float64}, fix_costs::AbstractMatrix{Float64})
    total_var_costs = 0
    total_fix_costs = 0
    for ((i, j), val) in sol
        if val > 0
            total_var_costs += var_costs[i, j] * val
            total_fix_costs += fix_costs[i, j]
        end
    end
    return total_var_costs + total_fix_costs
end


function sol_dict_to_matrix(sol::Dict{Tuple{Int,Int},Int}, m::Int, n::Int)
    sol_matrix = zeros(Int, (m, n))
    for ((i, j), val) in sol
        sol_matrix[i, j] = val
    end
    return sol_matrix
end

""" Print to log file. """
function write_to_log(filepath, item)
    open(filepath, "a") do file
        println(file, item)
    end
end

#############################################
# EA helper functions
#############################################

""" Convert cycle given by nodes into loop defined by edges. """
function node_cycle_to_edge_loop(cycle, m)
    loop = Tuple{Int,Int}[]
    for (i, v1) in enumerate(cycle)
        j = i % length(cycle) + 1
        v2 = cycle[j]
        edge = v1 < v2 ? (v1, v2 - m) : (v2, v1 - m)
        push!(loop, edge)
    end
    return loop
end

""" Reorder loop such that it starts with the entering variable. """
function reorder_loop(loop, var_in)
    index = findfirst(x -> x == var_in, loop)
    if index === nothing
        return loop
    end
    return circshift(loop, -(index - 1))
end

end