{%- extends 'null.tpl' -%}

{%- block header -%}
#!/usr/bin/env python
# coding: utf-8
{% endblock header %}

{% block in_prompt %}{% endblock in_prompt %}{% block input %}{{ cell.source | ipython2python }}
{% endblock input %}
{% block markdowncell scoped %}
{% endblock markdowncell %}
