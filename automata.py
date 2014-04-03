#!/usr/bin/env python3

# TODO list {{{1
# 1. more featrues in existing classes
# 2. better class hirarchy with good methods
# 3. pushdown automaton
# 4. docstrings
# 5. https://en.wikipedia.org/wiki/Stack_machine ???

# imports {{{1
import argparse
import cmd
import json
import os
import re
import sys
import glob


def curry(dictionary): #{{{1
    '''Take a dict with key type tuple and return a dict of dicts.

    See uncurry() for the inverse function.
    '''
    new = {}
    for key, val in dictionary.items():
        if type(key) == tuple:
            if new.__contains__(key[0]):
                new[key[0]][key[1]] = val
            else:
                new[key[0]] = {key[1]: val}
        else:
            raise TypeError(key)
    return new

def uncurry(dictionary): #{{{1
    '''Take a dict of dicts and return a dict with key type tuple.

    See curry() for the inverse function.
    '''
    new = {}
    for k1 in dictionary:
        for k2 in dictionary[k1]:
            new[(k1, k2)] = dictionary[k1][k2]
    return new

def parse_filename(filename): #{{{1
    '''Look at the filename to guess what type of data it contains.'''
    if filename[-4:] == '.xml':
        return 'xml'
    elif filename[-5:] == '.json':
        return 'json'
    elif filename[-4:] == '.txt':
        return 'text'
    else:
        raise ValueError('The file should have one of the following '
                'extensions: .txt, .json, .xml')

# copied from http://stackoverflow.com/questions/16826172
def complete_path(text, line): #{{{1
    arg = line.split()[1:]
    dir, base = '', ''
    try:
        dir, base = os.path.split(arg[-1])
    except:
        pass
    cwd = os.getcwd()
    try:
        os.chdir(dir)
    except:
        pass
    ret = [f+os.sep if os.path.isdir(f) else f for f in os.listdir('.') if
            f.startswith(base)]
    if base == '' or base == '.':
        ret.extend(['.' + os.path.sep, '..' + os.path.sep])
    elif base == '..':
        ret.append('..' + os.path.sep)
    os.chdir(cwd)
    return ret

class RegisterMachineCommand(): #{{{1

    # constants {{{2
    ZERO = 0
    INC = 1
    IF = 2
    zero_re = re.compile(r'^(x_?)?(\d+)\s*:?=\s*0')
    inc_re = re.compile(
            r'^(x_?)?(\d+)\s*(\+\+|\+=\s*1|:?=\s*(x_?)?\2\s*\+\s*1)')
    if_re = re.compile(r'^if\s*(x_?)?(\d+)\s*==?\s*(x_?)?(\d+)\s*then' +
            r'\s*(\d+)\s*else\s*(\d+)')

    # variables {{{2
    type = None
    register = None
    jumps = None

    def __init__(self, arg): #{{{2
        if type(arg) is str:
            self.parse_string(arg)
        elif type(arg) is list:
            self.parse_list(arg)
        else:
            raise TypeError()

    def __str__(self): #{{{2
        if self.type == self.ZERO:
            return 'x' + str(self.register) + ' = 0'
        elif self.type == self.INC:
            return 'x' + str(self.register) + ' = x' + str(self.register) + \
                    ' + 1'
        elif self.type == self.IF:
            return ('if x' + str(self.register[0]) + ' = x' +
                    str(self.register[1]) + ' then ' + str(self.jumps[0]) +
                    ' else ' + str(self.jumps[1]))
        else:
            return False

    def parse_string(self, arg): #{{{2
        match = self.zero_re.match(arg)
        if match is not None:
            self.type = self.ZERO
            self.register = int(match.group(2))
            return
        match = self.inc_re.match(arg)
        if match is not None:
            self.type = self.INC
            self.register = int(match.group(2))
            return
        match = self.if_re.match(arg)
        if match is not None:
            self.type = self.IF
            self.register = (int(match.group(2)), int(match.group(4)))
            self.jumps = (int(match.group(5)), int(match.group(6)))
            return
        raise Exception('Error parsing string: ' + arg)

    def parse_list(self, arg): #{{{2
        if arg[0] == self.ZERO or arg[0] == self.INC:
            self.type = arg[0]
            if type(arg[1]) is not int:
                raise TypeError(arg[1])
            else:
                self.register = arg[1]
        elif arg[0] == self.IF:
            self.type = self.IF
            if type(arg[1]) is int and type(arg[2]) is int and \
                    type(arg[3]) is int and type(arg[4]) is int:
                self.register = (arg[1], arg[2])
                self.jumps = (arg[3], arg[4])
            else:
                raise TypeError()
        else:
            raise ValueError(arg[0])


class Machine(): #{{{1
    '''The base class for all theoretical macines.'''

    def __init__(self, data=None, _type=None): #{{{2
        '''Initialize the machine by parsing data according to _type.

        Both arguments are strings.  _type is one of 'text', 'json', 'xml'.
        data is the serialized representation of the machine in the
        corresponding format.
        '''
        if data is not None and _type is not None:
            if _type == 'text':
                self.from_text(data)
            elif _type == 'json':
                self.from_json(data)
            elif _type == 'xml':
                self.from_xml(data)
        elif data is not None or _type is not None:
            raise ValueError('You must give both data and type or none.')

    # References to be implemented in subclass. {{{2
    def to_text(self): raise NotImplementedError()
    def to_json(self): raise NotImplementedError()
    def to_xml(self): raise NotImplementedError()
    def from_text(self, data): raise NotImplementedError()
    def from_json(self, data): raise NotImplementedError()
    def from_xml(self, data): raise NotImplementedError()
    def run(self, input): raise NotImplementedError()


class StateMachine(Machine): #{{{1

    # variables #{{{2
    table = {}
    alphabet = set()
    start_state = 0
    final_states = set()

    def __init__(self, **kwarg): #{{{2
        super(StateMachine, self).__init__(**kwarg)

    def set_start_state(self, state): #{{{2
        self.start_state = int(state)

    def set_final_states(self, states): #{{{2
        self.final_states = set([int(s) for s in states])

    def add_final_state(self, state): #{{{2
        self.final_states.add(int(state))

    def del_final_state(self, state): #{{{2
        self.final_states.difference_update([int(state)])

    def del_final_states(self, states=None): #{{{2
        if states is None:
            self.final_states.clear()
        else:
            self.final_states.difference_update([int(s) for s in states])

    def set_alphabet(self, alph): #{{{2
        'Set the alphabet, remove whitespace.'
        self.alphabet = set([str(x) for x in alph])
        self.alphabet.difference_update(list('\n\t\r\v '))

    def add_to_alphabet(self, alph): #{{{2
        self.alphabet.update([str(x) for x in alph])
        self.alphabet.difference_update(list('\n\t\r\v '))

    def del_from_alphabet(self, alph): #{{{2
        self.alphabet.difference_update(list(alph))

    def del_alphabet(self): #{{{2
        self.alphabet = {self.blank}

    def set_command(self, state1, field, state2): #{{{2
        self.table[(int(state1), str(field))] = int(state2)

    def del_command(self, state, field): #{{{2
        try:
            self.table.pop((int(state), str(field)))
        except KeyError:
            return False
        else:
            return True

    def run(self, tape, callback): #{{{2
        state = 0
        head = 0
        tape = list(tape)
        callback(self.format_position(tape, state, head))
        while True:
            try:
                field = tape[head]
            except IndexError:
                self.result = state in self.final_states
                return
            try:
                new_state = self.table[(state, field)]
            except KeyError:
                self.result = False
                return
            state = new_state
            head += 1
            callback(self.format_position(tape, state, head))

    @staticmethod
    def format_position(tape, state, head): #{{{2
        return ''.join(tape) + '\n' + ' ' * head + '^ ' + str(state)

    def to_text(self, verbose=False): #{{{2
        '''Produce a plain text representation of self, which can be parsed
        back by from_text and is human readable.  If verbose=True the result
        will contain descriptive titles for the data.'''
        if verbose:
            string = 'alphabet: '
        else:
            string = ''
        string += ''.join(sorted(self.alphabet)) + '\n'
        if verbose:
            string += 'start state: '
        string += str(self.start_state) + '\n'
        if verbose:
            string += 'final states: '
        string += ' '.join([str(x) for x in sorted(self.final_states)]) + '\n'
        if verbose:
            string += 'table:\n'
        for (state1, field), val in sorted(self.table.items()):
            string += str(state1) + ' ' + field
            # This is a little more general than actually needed here because
            # then this function can also be used for TuringMachine.
            if type(val) is list or type(val) is tuple:
                for x in val:
                    string += ' ' + str(x)
            else:
                string += ' ' + str(val)
            string += '\n'
        return string.strip('\n')

    def from_text(self, data): #{{{2
        'Parse some data which should be created with to_string.'
        alph, data = data.strip('\n').split('\n', 1)
        if alph.lower().startswith('alphabet: '):
            alph = alph.split(None, 1)[1]
        self.set_alphabet(alph)
        start, data = data.split('\n', 1)
        if start.lower().startswith('start state: '):
            start = start.split()[2]
        self.set_start_state(start)
        final, data = data.split('\n', 1)
        if final.lower().startswith('final states: '):
            final = final.split()[2:]
        else:
            final = final.split()
        self.set_final_states(final)
        data = data.strip('\n').split('\n')
        if data[0].lower() == 'table:':
            data = data[1:]
        for line in data:
            self.set_command(*line.split())

    def to_json(self): #{{{2
        data = self.__dict__.copy()
        data['table'] = curry(self.table)
        data['alphabet'] = sorted(self.alphabet)
        data['final_states'] = sorted(self.final_states)
        data['table'] = curry(self.table)
        return json.dumps(data, sort_keys=True)

    def from_json(self, data): #{{{2
        data = json.loads(data)
        self.set_alphabet(data['alphabet'])
        self.set_start_state(data['start_state'])
        self.set_final_states(data['final_states'])
        # This is a little more general than actually needed here because then
        # this function can also be used for TuringMachine.
        for key, val in uncurry(data['table']).items():
            arg = list(key)
            if type(val) is list or type(val) is tuple:
                arg.extend(val)
            else:
                arg.append(val)
            self.set_command(*arg)

        def _un_reachable_states(): #{{{2
            # this algorithm is from
            # http://en.wikipedia.org/wiki/DFA_minimization
            #let reachable_states:= {q0};
            #let new_states:= {q0};
            #do {
            #    temp := the empty set;
            #    for each q in new_states do
            #        for all c in ∑ do
            #            temp := temp ∪ {p such that p=δ(q,c)};
            #        end;
            #    end;
            #    new_states := temp \ reachable_states;
            #    reachable_states := reachable_states ∪ new_states;
            #} while(new_states ≠ the empty set);
            #unreachable_states := Q \ reachable_states;
            reachable = {self.start_state}
            new = {self.start_state}
            while new != set():
                tmp = set()
                for state in new:
                    tmp.update([self.tape[(state, alph)] for alph in self.alphabet])
                new = tmp.difference(reachable)
                reachable.update(new)
            states = set(self.table)
            states.update([val for key, val in self.table.items()])
            return (states.difference(reachable), reachable)


class PushDownMachine(StateMachine): #{{{1

    # constants {{{2
    BOTTOM = '_'

    # variables {{{2
    stack = [BOTTOM]
    stack_alphabet = set([BOTTOM])

    def __init__(self, **kwarg): #{{{2
        super(PushDownMachine, self).__init__(**kwarg)

    def pop(self): #{{{2
        try:
            return self.stack.pop()
        except IndexError:
            return self.BOTTOM

class TuringMachine(StateMachine): #{{{1

    # constants (FIXME should be static) {{{2
    #left = '\u001c'
    #right = '\u001d'
    BLANK = '_'
    LEFT = '<<'
    RIGHT = '>>'

    # variables {{{2
    alphabet = set()

    def __init__(self, **kwarg): #{{{2
        super(TuringMachine, self).__init__(**kwarg)

    @staticmethod
    def action_to_string(action): #{{{2
        '''At the moment action is realized as a string, so this method does
        nothing.'''
        #if action == cls.LEFT:
        #    return cls.moveLeft
        #elif action == cls.RIGHT:
        #    return cls.moveRight
        #else:
        #    return str(action)
        return action

    def set_alphabet(self, alph): #{{{2
        'Set the alphabet, remove whitespace and add the blank character.'
        self.alphabet = set([str(x) for x in alph])
        self.alphabet.difference_update(list('\n\t\r\v '))
        self.alphabet.add(self.BLANK)

    def del_from_alphabet(self, alph): #{{{2
        self.alphabet.difference_update(list(alph))
        self.alphabet.add(self.blank)

    def set_command(self, state1, field, action, state2): #{{{2
        self.table[(int(state1), str(field))] = (str(action), int(state2))

    def run(self, tape, callback): #{{{2
        start = 0
        state = 0
        head = 0
        tape = list(tape)
        if tape == []:
            tape = [self.BLANK]
        callback(self.format_position(tape, state, start, head))
        while True:
            field = tape[head]
            try:
                action, new_state = self.table[(state, field)]
            except KeyError as e:
                self.result = state in self.final_states and head == start
                return
            if action == self.LEFT:
                if head == 0:
                    tape.insert(0, self.BLANK)
                    start = start + 1
                else:
                    head = head - 1
            elif action == self.RIGHT:
                head = head + 1
                if head == len(tape):
                    tape.append(self.BLANK)
            else:
                tape[head] = action
            state = new_state
            callback(self.format_position(tape, state, start, head))

    @staticmethod
    def format_position(tape, state, start, head): #{{{2
        string = ''.join(tape) + '\n'
        if start == head:
            string += ' ' * head
            string += 'T'
        elif start < head:
            string += ' ' * start
            string += '|'
            string += ' ' * (head - start - 1)
            string += '^'
        else:
            string += ' ' * head
            string += '^'
            string += ' ' * (start - head - 1)
            string += '|'
        return string + ' ' + str(state)


class RegisterMachine(Machine): #{{{1
    '''A register machine.

    See http://www.math.lmu.de/~schwicht/lectures/logic/ws12/ch3.pdf for the
    matehemaical background.
    '''

    # constants {{{2

    # variables {{{2
    registers = {}
    commands = []

    def __init__(self, **kwarg): #{{{2
        super(RegisterMachine, self).__init__(**kwarg)

    def to_text(self, verbose=False): #{{{2
        return '\n'.join([str(c) for c in self.commands])


    def from_text(self, data): #{{{2
        for line in data.strip('\n').split('\n'):
            self.add(RegisterMachineCommand(line))

    def add(self, command): #{{{2
        if type(command) == str:
            command = RegisterMachineCommand(command)
        self.commands.append(command)

    def run(self, input): #{{{2
        self.registers = {}
        self.registers.update(enumerate(input))
        current = 0
        while len(self.commands) > current:
            command = self.commands[current]
            if command.type == RegisterMachineCommand.ZERO:
                self.registers[command.register] = 0
                current += 1
            elif command.type == RegisterMachineCommand.INC:
                self.registers[command.register] += 1
                current += 1
            elif command.type == RegisterMachineCommand.IF:
                if self.registers[command.register[0]] == \
                        self.registers[command.register[1]]:
                    current = command.jumps[0]
                else:
                    current = command.jumps[1]
        self.result = self.registers[0]

    def print_registers(self): #{{{2
        for reg, val in self.registers.items():
            print('x' + str(reg) + ' = ' + str(val))


class CLI(cmd.Cmd): #{{{1

    # constants {{{2
    intro = ('Welcome to the interactive session where you can define and'
            ' run several types of machines.  Type help or ? for a list of'
            ' available commands.\n')
    prompt = '(automata) '

    def __init__(self, machine=None): #{{{2
        super(CLI, self).__init__()
        if machine is not None:
            self.machine = machine
        else:
            self.machine = self.machine_type()
        self.last = True
        self.run = True

    def do_print(self, args): #{{{2
        '''
        Print the current machine.  PRINT
        '''
        print(self.machine.to_text(verbose=True))
    do_p = do_print

    def do_dump(self, args): #{{{2
        #import pdb; pdb.set_trace()
        print(self.machine.__dict__)
        print(self.machine.to_json())

    def do_load(self, args): #{{{2
        '''
        Load a machine from a file.  LOAD path/filename.ext
        '''
        string = ''
        args = args.strip()
        try:
            _type = parse_filename(args)
        except ValueError as e:
            print(e)
            return
        with open(args) as fd:
            self.machine = self.machine_type(_type=_type, data=fd.read())

    def complete_load(self, text, line, begidx, endidx): #{{{2
        return complete_path(text, line)

    def do_clear(self, args): #{{{2
        '''
        Clear the current machine starting fresh.  CLEAR
        '''
        if args is not None and args.strip() != '':
            print('This command does not take arguments!')
            return
        else:
            self.machine = self.machine_type()

    def do_run(self, args): #{{{2
        '''
        Run the current machine on the supplied input.  RUN some input ...
        '''
        ret = self.machine.run(self.prepare_input_for_run(args), print)
        if type(self.machine.result) is bool:
            if ret:
                print('Machine terminated sucessfull.')
            else:
                print('Machine failed.')
        else:
            print('Result:', self.machine.result)
    do_r = do_run

    def do_save(self, args): #{{{2
        '''
        Save the current machine to a file.  SAVE path/filename
        '''
        args = args.strip()
        if args[-4:] == '.xml':
            data = self.machine.to_xml()
        elif args[-5:] == '.json':
            data = self.machine.to_json()
        elif args[-4:] == '.txt':
            data = self.machine.to_text(verbose=True)
        else:
            print('Please give one of the following file extensions:')
            print('.txt', '.json', '.xml')
            return
        try:
            with open(args, 'w') as fd:
                fd.write(data)
        except PermissionError:
            print('Error: Could not open file', args, 'for writing.')
            return
        # TODO which other errors are possible?
        # FileNotFoundError

    def complete_save(self, text, line, begidx, endidx): #{{{2
        return complete_path(text, line)

    def do_switch(self, args): #{{{2
        '''
        Switch to another type of machine.  There will be other commands
        available.  The current machine will be deleted.
            SWITCH machine type
        '''
        args = args.strip().lower()
        for cls in CLI.__subclasses__():
            if cls.__name__.lower() == args:
                # FIXME, this seems ugly:
                # We use a global variable next_cli_class to communicate the
                # switch to the main program.
                #import pdb; pdb.set_trace()
                global next_cli_class
                next_cli_class = cls
                return True

    def complete_switch(self, text, line, begidx, endidx): #{{{2
        classes = set(CLI.__subclasses__())
        changed = True
        while changed:
            new = set()
            for cls in classes:
                new.update(cls.__subclasses__())
                new.difference_update(classes)
            classes.update(new)
            changed = new == set()
        return [cls.__name__ for cls in classes if
                cls.__name__.lower().startswith(text.lower())]

    def do_shell(self, args): #{{{2
        '''
        Execute an external shell command.  SYSTEM shell-commands
        '''
        os.system(args)
    do_system = do_shell

    def do_quit(self, args): #{{{2
        '''
        Quit the program. Return the return value of the last run as
        exit code.  QUIT [exit code]
        '''
        if args.strip() is not '':
            try:
                ret = int(args)
            except:
                ret = -1
        elif self.last:
            ret = 0
        else:
            ret = 1
        sys.exit(ret)
    do_q = do_quit

    def do_EOF(self, args): #{{{2
        '''
        Same as quit.  You can press CTRL-D to stop this horror.
        '''
        self.do_quit(args)


class StateMachineCLI(CLI): #{{{1

    # constants {{{2
    prompt = '(fsm) '
    machine_type = StateMachine

    def do_add(self, args): #{{{2
        '''
        Add an instruction to the current machine.  ADD instruction
        '''
        try:
            s, k, n = args.split()
        except:
            print('Error: wrong arguments (exactly three are needed)')
        else:
            self.machine.set_command(s, k, n)

    def do_alph(self, args): #{{{2
        '''
        Set the alphabet of the current machine.  ALPH  thenewalphabet
        '''
        self.machine.set_alphabet(args)

    def do_final(self, args): #{{{2
        '''
        Set the final states for the machine from the arguments.
        '''
        self.machine.set_final_states(args.split())

    def do_remove(self, args): #{{{2
        '''
        Remove a single command from the table of the machine.  To remove a
        command mapping state_1 and field_1 to action_2 and state_2 specify
        state_1 and field_1.
        '''
        state, field = args.split()
        state = int(state)
        if not self.machine.del_command(state, field):
            print('Such a command was not present.')

    @staticmethod
    def prepare_input_for_run(arg): #{{{2
        'Do not change the input.'
        return arg


class TuringCLI(StateMachineCLI): #{{{1

    # constants {{{2
    prompt = '(tm) '
    machine_type = TuringMachine

    def do_add(self, args): #{{{2
        '''
        Add an instruction to the current machine.  ADD instruction
        '''
        try:
            s, k, a, n = args.split()
        except:
            print('Error: wrong arguments (exactly four are needed)')
        else:
            self.machine.set_command(s, k, a, n)


class RegisterCLI(CLI): #{{{1

    # constants {{{2
    prompt = '(rm) '
    machine_type = RegisterMachine

    @staticmethod
    def prepare_input_for_run(args): #{{{2
        'Turn the input into a list of integers.'
        return [int(x) for x in args.split()]


if __name__ == '__main__': #{{{1

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=argparse.FileType('r'),
            help='a file to load a machine from')
    parser.add_argument('-i', '--input', help='the input to process')
    parser.add_argument('-t', '--type',
            help='the type of machine to simulate',
            choices=['turing', 'register', 'state'], default='turing')
    args = parser.parse_args()

    if args.type == 'turing':
        cli = TuringCLI()
    elif args.type == 'register':
        cli = RegisterCLI()
    elif args.type == 'state':
        cli = StateMachineCLI()

    if args.file is not None:
        _type = parse_filename(args.file.name)
        cli.machine = cli.machine_type(_type=_type, data=args.file.read())
    if args.input is not None:
        if args.file is None:
            parser.error('If you specify some input you must also specify a '
                    'machine file.')
        else:
            # run directly and exit
            cli.do_run(args.input)
            sys.exit()

    while True:
        #FIXME, this seems ugly:
        # We use this global varibale to communicate a switch from inside the
        # cli class.
        next_cli_class = None
        cli.cmdloop()
        if next_cli_class is None:
            sys.exit()
        else:
            cli = next_cli_class()

# vim: foldmethod=marker
