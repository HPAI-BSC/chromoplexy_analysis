

def load_breaks(file_location):
    '''
    Reads a ".vcf.tsv" file, storing the breaks in a dictionary.
    Input:
        file_location: complete path to "vcf.tsv" file, containing breaks in a genome
    Output:
        breaks_by_chromosome: dictionary {str:[int]}, where keys are chromosome ids
            and the corresponding list contains the position of breaks in that 
            chromosome (sorted).
        list_of_pairs: list[((str,int),(str,int))]. List of breaks, each entry contains
            first chromosome and position within, second chromosome and position within
    '''
    breaks_by_chromosome = {}
    list_of_pairs = []
    #print 'Reading file:',file_location
    with open(file_location) as f:
        #Skip the first line, which is a descriptor of fields
        f.next()
        #Read break by break
        for l in f:
            if len(l.split('\t'))!=5:
                raise Exception("Wrong number of fields (i.e., not 5) in line",l)
            chromosome1, chromosome1_pos, chromosome2, chromosome2_pos, break_type = l.split('\t')
            chromosome1_pos = int(chromosome1_pos)
            chromosome2_pos = int(chromosome2_pos)
            #If its the first break found in this chromosome, initialize the chromosome list
            if chromosome1 not in breaks_by_chromosome.keys():
                breaks_by_chromosome[chromosome1] = []
            #Store the break in the corresponding dictionary entry of the chromosome
            breaks_by_chromosome[chromosome1].append(chromosome1_pos)
            #The same for the second chromosome
            if chromosome2 not in breaks_by_chromosome.keys():
                breaks_by_chromosome[chromosome2] = []
            breaks_by_chromosome[chromosome2].append(chromosome2_pos)
            list_of_pairs.append(((chromosome1, chromosome1_pos),
                    (chromosome2, chromosome2_pos)))
    #Sort the lists
    for k in breaks_by_chromosome.keys():
        breaks_by_chromosome[k] = sorted(breaks_by_chromosome[k],key=int)
    return breaks_by_chromosome, list_of_pairs
