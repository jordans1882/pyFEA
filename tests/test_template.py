"""Provides some tests of the template class.

Includes test of constructor.
Includes test of identity function.
"""

from feareu.template import Template


def test_template_attributes():
    """Test template attribute defaults."""
    template = Template()
    assert template.val == 42


def test_template_identity_fun():
    """Test template identity function."""
    template = Template()
    tval = template.identity(42.31)
    assert tval == 42.31
