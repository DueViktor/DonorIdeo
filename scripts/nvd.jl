using CSV, DataFrames, Laplacians, LinearAlgebra, Dates


# check if data/nvd/ exists, if not, create it
if !isdir("data/nvd")
    mkdir("data/nvd")
end

G = readIJ("data/temporary/nvd-network.csv"); # The network MUST be comma separated, with nodes as numerical indexes from 1 to n, without gaps

df = DataFrame(CSV.File("data/temporary/nvd-node_attributes.csv", delim=',', ignorerepeated=true))
df_names = names(df)
size_of_names = size(df_names)[1]

solver = approxchol_lap(G);

# iterate over the names of the dataframe except the first column
for i in range(2, size_of_names)

    output_file = "data/nvd/$(df_names[i]).csv"

    # If the file already exists, skip this iteration
    if isfile(output_file)
        continue
    end

    # Create the file and write the timestamp
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    open(output_file, "a") do f
        write(f, "$timestamp\n")
    end

    for j in range(2, size_of_names)

        # The distance is symmetric, so we only need to calculate it once
        if i != j && i < j
            o = df[:, df_names[i]] - df[:, df_names[j]]
            x = solver(o)
            ge = sqrt(dot(o', x)) # This is the number you want

            # save the results to a file
            open(output_file, "a") do f
                write(f, "$(df_names[i]),$(df_names[j]),$ge\n")
            end
        end
    end
end
