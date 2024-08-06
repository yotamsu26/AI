import sys

def create_propositions(disks, pegs):
    propositions_string = "Propositions: \n"
    propositions_string += write_disk_on_disk(disks)
    for disk in disks:
        for peg in pegs:
            propositions_string += write_object1_on_object2(disk, peg)
            
    for disk in disks:
        propositions_string += write_free_object(disk)
            
    for peg in pegs:
        propositions_string += write_free_object(peg)
        
    propositions_string += '\n'
            
    return propositions_string

def create_action(src, dist, moving_disk):
    action_string = ''
    
    name = f'\nName: {moving_disk}_FROM_{src}_TO_{dist}'
    action_string += name

    pre = '\npre: ' + write_free_object(moving_disk) + write_free_object(dist) + write_object1_on_object2(moving_disk, src)
    action_string += pre
    
    add = '\nadd: ' + write_object1_on_object2(moving_disk, dist) + write_free_object(src)
    action_string += add
        
    delete = '\ndelete: ' + write_free_object(dist) + write_object1_on_object2(moving_disk, src)
    action_string += delete
    
    return action_string
        
    
def create_actions(disks, pegs):
    actions_string = "Actions: "
    
    # disk to peg, possible all disks to all pegs, disk must be on wider disk
    # also peg to disk
    for disk1 in range(len(disks)):
        for disk2 in range(disk1 + 1, len(disks)):
            for peg in pegs:
                actions_string += (create_action(disks[disk2], peg, disks[disk1]))
                actions_string += (create_action(peg, disks[disk2], disks[disk1]))
            
    # from peg to other peg
    for peg1 in pegs:
        for peg2 in pegs:
            if peg1 != peg2:
                for disk in disks:
                    actions_string += (create_action(peg1, peg2, disk))
                    
    # from disk to disk
    for disk1 in range(len(disks)):
        for disk2 in range(disk1 + 1, len(disks)):
            for disk3 in range(disk1 + 1, len(disks)):
                if disk2 != disk3:
                    actions_string += (create_action(disks[disk2], disks[disk3], disks[disk1]))
    
    return actions_string

def create_domain_file(domain_file_name, n_, m_):
    disks = ['d_%s' % i for i in list(range(n_))]  # [d_0,..., d_(n_ - 1)]
    pegs = ['p_%s' % i for i in list(range(m_))]  # [p_0,..., p_(m_ - 1)]
    domain_file = open(domain_file_name, 'w')  # use domain_file.write(str) to write to domain_file
    "*** YOUR CODE HERE ***"
    domain_file.write(create_propositions(disks, pegs))
    domain_file.write(create_actions(disks, pegs))

    domain_file.close()
    
def write_free_object(object):
    return f'f_{object} '

def write_disk_on_disk(disks):
    disk_string = ""
    for d1 in range(len(disks)):
        for d2 in range(d1+1, len(disks)):
            disk_string += f'{disks[d1]}_on_{disks[d2]} '
            
    return disk_string

def write_object1_on_object2(object1, object2):
    return f'{object1}_on_{object2} '

def initial_state_template(disks, pegs):
    initnal_state_string = f'Initial state: '
    
    # all the disks on the first peg
    for disk in range(0, len(disks)-1):
        initnal_state_string += write_object1_on_object2(disks[disk], disks[disk+1])
    initnal_state_string += (write_object1_on_object2(disks[-1], pegs[0]))
    
    # all the free state for the actions
    initnal_state_string += (write_free_object(disks[0]))
    for i in range(1, len(pegs)):
        initnal_state_string += (write_free_object(pegs[i]))
        
    initnal_state_string += '\n'
        
    return initnal_state_string

def goal_state_template(disks, pegs):
    goal_state_string = f'Goal state: '
    # all the disks on the last peg
    for disk in range(0, len(disks)-1):
        goal_state_string += write_object1_on_object2(disks[disk], disks[disk+1])
    goal_state_string += (write_object1_on_object2(disks[-1], pegs[-1]))
    
    return goal_state_string

def create_problem_file(problem_file_name_, n_, m_):
    disks = ['d_%s' % i for i in list(range(n_))]  # [d_0,..., d_(n_ - 1)]
    pegs = ['p_%s' % i for i in list(range(m_))]  # [p_0,..., p_(m_ - 1)]
    problem_file = open(problem_file_name_, 'w')  # use problem_file.write(str) to write to problem_file
    "*** YOUR CODE HERE ***"
    problem_file.write(initial_state_template(disks, pegs))
    problem_file.write(goal_state_template(disks, pegs))

    problem_file.close()


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: hanoi.py n m')
        sys.exit(2)

    n = int(float(sys.argv[1]))  # number of disks
    m = int(float(sys.argv[2]))  # number of pegs

    domain_file_name = 'hanoi_%s_%s_domain.txt' % (n, m)
    problem_file_name = 'hanoi_%s_%s_problem.txt' % (n, m)

    create_domain_file(domain_file_name, n, m)
    create_problem_file(problem_file_name, n, m)
