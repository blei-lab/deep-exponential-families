"""
  This script helps convert from our LDA-C format to the DEF input
  format. 

  Our LDA-C format:

    A text file where each line is of the form:

    [lineNumber] [M] [term_1]:[count] [term_2]:[count] ... [term_N]:[count]

    where [M] is the number of unique terms in the document, and the [count]
    associated with each term is how many times that term appeared in the
    document. [lineNumber] is a number indicating the corresponding line
    number in the file (starts at 0). 

  Converting data: 
    Converting train data:
    python convert_data.py --trainfile train.dat                                           --outfile train.cpp.dat

    Convertaing validation/test data: 
    python convert_data.py --trainfile validation-train.dat --testfile validation-test.dat --outfile validation.cpp.dat
    python convert_data.py --trainfile test-train.dat       --testfile test-test.dat       --outfile test.cpp.dat
"""


import argparse
from collections import defaultdict

def read_from_dat(fname):
    r_list, c_list, v_list = [], [], []
    with open(fname) as ifile:
        for l in ifile:
            ele = l.strip().split()
            ind = int(ele[0])
            ct = int(ele[1])
            assert ct == len(ele)-2
            wc_list = map(lambda p: map(int, p.split(':')), ele[2:])

            for c, v in wc_list:
                r_list.append(ind)
                c_list.append(c)
                v_list.append(v)
    print 'read from %s, max_row_ind %s, max_col_ind %d' % (
        fname, max(r_list), max(c_list))

    return r_list, c_list, v_list

# read from old data format without the row-indices
def read_from_old_dat(fname):
    r_list, c_list, v_list = [], [], []
    with open(fname) as ifile:
        ind = 0
        header = False
        for l in ifile:
            if header:
                header = False
                continue

            ele = l.strip().split()
            ct = int(ele[0])
            assert ct == len(ele)-1, l
            wc_list = map(lambda p: map(int, p.split(':')), ele[1:])
            for c, v in wc_list:
                r_list.append(ind)
                c_list.append(c)
                v_list.append(v)
            ind += 1
    print 'read from %s, max_row_ind %s, max_col_ind %d' % (
        fname, max(r_list), max(c_list))

    return r_list, c_list, v_list

def write_to_cpp_dat(fname, r_list, c_list, v_list):
    print 'write to %s, max_row_ind %d, max_column_ind %d' % (
        fname, max(r_list), max(c_list))
    m = defaultdict(dict)
    for r, c, v in zip(r_list, c_list, v_list):
        m[r][c] = v

    with open(args.outfile, "w") as ofile:
        ofile.write('%d %d\n' % (max(r_list)+1, max(c_list)+1))
        # we output 0 rows
        for r in range(0, max(r_list)+1):
            ofile.write('%d %d\n' % (r, len(m[r])))
            ofile.write(' '.join(map(lambda (c, v): '%d %d' % (c, v),
                                     sorted(m[r].items()))))
            ofile.write('\n')                            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process input for C++ to read')
    parser.add_argument('--trainfile', default=None)
    parser.add_argument('--testfile', default=None)
    parser.add_argument('--outfile', default=None)
    parser.add_argument('--trans', default=False, action='store_true')
    parser.add_argument('--old_dat', default=False, action='store_true')
    args = parser.parse_args()
    print args

    read_proc = read_from_old_dat if args.old_dat else read_from_dat
    r_list, c_list, v_list = read_proc(args.trainfile)
    if args.testfile:
        tr, tc, tv = read_proc(args.testfile)
        r_list += tr
        c_list += tc
        v_list += map(lambda v: -v-1, tv)

    if args.trans:
        write_to_cpp_dat(args.outfile, c_list, r_list, v_list)
    else:
        write_to_cpp_dat(args.outfile, r_list, c_list, v_list)
