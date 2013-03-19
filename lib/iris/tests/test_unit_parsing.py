# -*- coding: utf-8 -*-

if __name__ == '__main__':

    examples = [
                ('s', '[s]', '\mathrm{s}'),
                ('m per second', '[(m / second)]', r'\frac{\mathrm{m}}{\mathrm{second}}'),
                ('meter second-1', '[(meter * (second ^ -1))]', r'\mathrm{meter}\ \mathrm{second}^{-1}'),
                ('m pER second', '[(m / second)]', r'\frac{\mathrm{m}}{\mathrm{second}}'),
                ('2*s^2', '[(2 * (s ^ 2))]', r'{2}\mathrm{s}^{2}'),
                ('2s2', '[(2 * (s ^ 2))]', r'{2}\mathrm{s}^{2}'),
                ('s2.3', '[((s ^ 2) * 3)]', r'\mathrm{s}^{2}{3}'),
                ('s^s', '[(s ^ s)]', r'\mathrm{s}^\mathrm{s}'),
                ('s+2+2', '[((s ^ 2) * 2)]', r'\mathrm{s}^{2}{2}'),
                ('s1 s1', '[((s ^ 1) * (s ^ 1))]', r'\mathrm{s}^{1}\mathrm{s}^{1}'),
                ('2**2+2s2', '[(((2 ^ 2) * 2) * (s ^ 2))]', r'{2}^{2}{2}\mathrm{s}^{2}'),
                ('s-1+2', '[((s ^ -1) * 2)]', r'\mathrm{s}^{-1}{2}'),
                ('1/s+2', '[(1 / (s ^ 2))]', r'\frac{{1}}{\mathrm{s}^{2}}'),
                ('(s+2) * ( s2)', '[((s ^ 2) * (s ^ 2))]', r'\mathrm{s}^{2}\mathrm{s}^{2}'),
                ('s**2 s+2+2', '[((s ^ 2) * ((s ^ 2) * 2))]', r'\mathrm{s}^{2}\mathrm{s}^{2}{2}'),
                ('1 / s -2', '[((1 / s) * -2)]', r'\frac{{1}}{\mathrm{s}}({-2})'),
                ('1 / (s -2)', '[(1 / (s * -2))]', r'\frac{{1}}{\mathrm{s}\ ({-2})}'),
                ('second*s', '[(second * s)]', r'\mathrm{second}\ \mathrm{s}'),
                ('s·2', '[(s * 2)]', r'\mathrm{s}\ {2}'),
                (' -s2', '[(-s ^ 2)]', '\mathrm{-s}^{2}'),
                ('+s2', '[(s ^ 2)]', '\mathrm{s}^{2}'),
                ('+ s2', '[(s ^ 2)]', '\mathrm{s}^{2}'),
                ('-s2', '[(-s ^ 2)]', '\mathrm{-s}^{2}'),
                ('-s', '[-s]', '\mathrm{-s}'),
                ('(((a)* (b+2 ))/(b  2+2))', '[((a * (b ^ 2)) / (b * (2 ^ 2)))]', r'\frac{\mathrm{a}\ \mathrm{b}^{2}}{\mathrm{b}\ {2}^{2}}'),


                # The following are tested against UDUNITS and are *all* == s^2
                ('s ** 2', '[(s ^ 2)]', r'\mathrm{s}^{2}'),
                ('(s.s2).s-1', '[((s * (s ^ 2)) * (s ^ -1))]', r'\mathrm{s}\ \mathrm{s}^{2}\mathrm{s}^{-1}'),
                ('s/1/s-2', '[((s / 1) / (s ^ -2))]', r'\frac{\frac{\mathrm{s}}{{1}}}{\mathrm{s}^{-2}}'),
                ('s/(1/s1)', '[(s / (1 / (s ^ 1)))]', r'\frac{\mathrm{s}}{\frac{{1}}{\mathrm{s}^{1}}}'),
                ('(s/1)/s-1', '[((s / 1) / (s ^ -1))]', r'\frac{\frac{\mathrm{s}}{{1}}}{\mathrm{s}^{-1}}'),
                ('s+50 s-48', '[((s ^ 50) * (s ^ -48))]', r'\mathrm{s}^{50}\mathrm{s}^{-48}'),
                ('((s.2) / (2s)) (s2)', '[(((s * 2) / (2 * s)) * (s ^ 2))]', r'\frac{\mathrm{s}\ {2}}{{2}\mathrm{s}}\mathrm{s}^{2}'),
                ('s+2.+1.(-1).(-1)', '[((((s ^ 2) * 1.0) * -1) * -1)]', r'\mathrm{s}^{2}{1.0}({-1})({-1})'),
                ('s+4+2/(2s4) s2', '[((((s ^ 4) * 2) / (2 * (s ^ 4))) * (s ^ 2))]', r'\frac{\mathrm{s}^{4}{2}}{{2}\mathrm{s}^{4}}\mathrm{s}^{2}'),

                ('(2^2s)/4*s', '[((((2 ^ 2) * s) / 4) * s)]', r'\frac{{2}^{2}\mathrm{s}}{{4}}\mathrm{s}'),
                ('(s+2 2) / 2', '[(((s ^ 2) * 2) / 2)]', r'\frac{\mathrm{s}^{2}{2}}{{2}}'),

                # Some really subtle ones
                ('s +2', '[(s * 2)]', r'\mathrm{s}\ {2}'),
                ('s+2', '[(s ^ 2)]', r'\mathrm{s}^{2}'),
                ('s -2+3', '[(s * (-2 ^ 3))]', r'\mathrm{s}\ ({-2})^{3}'),
                ('s +     2', '[(s * 2)]', r'\mathrm{s}\ {2}'),

                # Some awkward numbers
                ('1', r'[1]', r'{1}'),
                ('1.', r'[1.0]', r'{1.0}'),
                ('1.2', r'[1.2]', r'{1.2}'),
                ('1.m', r'[(1.0 * m)]', r'{1.0}\mathrm{m}'),
                ('1.2e2m2', r'[(120.0 * (m ^ 2))]', r'{120.0}\mathrm{m}^{2}'),
                ('1e2m2', r'[(100.0 * (m ^ 2))]', r'{100.0}\mathrm{m}^{2}'),
                ('1e-2m2', r'[(0.01 * (m ^ 2))]', r'{0.01}\mathrm{m}^{2}'),
                ('1e+2m2', r'[(100.0 * (m ^ 2))]', r'{100.0}\mathrm{m}^{2}'),
                ('1.1+3', r'[(1.1 ^ 3)]', r'{1.1}^{3}'),
                ('1.1+3.1', r'[(1.1 * 3.1)]', r'{1.1}\times{3.1}'),
                ('1.2^3.1', r'[((1.2 ^ 3) * 0.1)]', r'{1.2}^{3}{0.1}'),
                ('m3.2', r'[((m ^ 3) * 2)]', r'\mathrm{m}^{3}{2}'),
                ('1.2+2.4+3.6', r'[((1.2 * 2.4) * 3.6)]', r'{1.2}\times{2.4}\times{3.6}'),
                ('1.2^2.4^3.6', r'[(((1.2 ^ 2) * (0.4 ^ 3)) * 0.6)]', r'{1.2}^{2}{0.4}^{3}{0.6}'),
                ('1.2+2.4^3.6', r'[((1.2 * (2.4 ^ 3)) * 0.6)]', r'{1.2}\times{2.4}^{3}{0.6}'),
                ('1.2^+2.4^3.6', r'[(((1.2 ^ 2) * (0.4 ^ 3)) * 0.6)]', r'{1.2}^{2}{0.4}^{3}{0.6}'),
                ('1.2*m2.3', r'[((1.2 * (m ^ 2)) * 3)]', r'{1.2}\mathrm{m}^{2}{3}'),
                ('4*8', '[(4 * 8)]', r'{4}\times{8}'),
                ('4*-8', '[(4 * -8)]', r'{4}({-8})'),
                ('4.+3', r'[(4.0 ^ 3)]', r'{4.0}^{3}'),
                ('-4.-3', r'[(-4.0 ^ -3)]', r'{-4.0}^{-3}'),
                ('3**2.5', r'[((3 ^ 2) * 0.5)]', r'{3}^{2}{0.5}'),
                ('3.0**2.5', r'[((3.0 ^ 2) * 0.5)]', r'{3.0}^{2}{0.5}'),
                ('(2 * 4) +-8', '[((2 * 4) * -8)]', r'{2}\times{4}({-8})'),
                ('(2 * (4 / 2)) +-8', '[((2 * (4 / 2)) * -8)]', r'{2}\frac{{4}}{{2}}({-8})'),
                ('((2-2 / 5) * 4) +-8', '[((((2 ^ -2) / 5) * 4) * -8)]', r'\frac{{2}^{-2}}{{5}}{4}({-8})'),
                ('-2 * -4 * -8', '[((-2 * -4) * -8)]', r'({-2})({-4})({-8})'),
                ('-2 * ((1*2) 4 -8)', '[(-2 * (((1 * 2) * 4) * -8))]', r'({-2}){1}\times{2}\times{4}({-8})'),
                ('(2 -1.) * --1', '[((2 * -1.0) * 1)]', r'{2}({-1.0}){1}'),

                # **Really** fruity examples
                ('s+2.+1.(-1.-1).-1', '[((((s ^ 2) * 1.0) * (-1.0 ^ -1)) * -1)]', r'\mathrm{s}^{2}{1.0}({-1.0})^{-1}({-1})'),
                ('s+2.+1.(-1.-1).-1.--1.--1', '[((((s ^ 2) * 1.0) * (-1.0 ^ -1)) * ((-1.0 ^ 1) * 1))]', r'\mathrm{s}^{2}{1.0}({-1.0})^{-1}{-1.0}^{1}{1}'),


                # Log units (NOT SUPPORTED)
#                ('lb(m^2)', r'', r''),
#                (),

                ]

    from iris.unit_tex import UnitParser
    UnitParser = UnitParser()

    for string_to_parse, expected_str, expected_tex in examples:
        print
        print string_to_parse
        tokenized = UnitParser.parse(string_to_parse)

        print tokenized
        tokenized_string = '[{!s}]'.format(tokenized)
        assert str(tokenized_string) == expected_str, 'Expected {}. Got {}.'.format(expected_str, tokenized_string)

        tokenized_latex = tokenized.latex()

        if tokenized_latex != expected_tex:
            import matplotlib.pyplot as plt
            import matplotlib

            with matplotlib.rc_context({'text.usetex': True, 'font.family': 'serif'}):
                plt.text(0.5, 0.7, r'{}'.format(string_to_parse), size=30, ha='center')
                if expected_tex:
                    plt.text(0.5, 0.5, r'${}$'.format(expected_tex), size=30, ha='center')
                plt.text(0.5, 0.2, r'${}$'.format(tokenized_latex), size=30, ha='center')

                plt.title('Failed with: {}'.format(string_to_parse))
            show = True
            if show: plt.show(block=False)
            print 'Expected {}. Got {}.'.format(expected_tex, tokenized_latex)
            if show: plt.show()
            exit()

        import iris.unit
        try:
            orig_unit = iris.unit.Unit(string_to_parse)
        except ValueError:
            # Don't worry about this unit if UDUNITS can't even parse it!
            continue

        simplified_unit = iris.unit.Unit(tokenized_string[1:-1].replace(' ', ''))

        print orig_unit == simplified_unit
