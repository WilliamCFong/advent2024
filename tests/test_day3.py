from advent import day3


def test_day3_part_1():
    sum_ = day3.find_product_sum(
        "xmul(2,4)%&mul[3,7]!@^do_not_mul(5,5)+"
        "mul(32,64]then(mul(11,8)mul(8,5))",
        day3.return_all_mul,
    )
    assert sum_ == 161


def test_day3_part_2():
    sum_ = day3.find_product_sum(
        "xmul(2,4)&mul[3,7]!^don't()_mul(5,5)+mul"
        "(32,64](mul(11,8)undo()?mul(8,5))",
        day3.yield_valid_mul,
    )
    assert sum_ == 48
