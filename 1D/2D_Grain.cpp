//  1D Grain proof of concept
//  built upon MFEM examples 1 & 27

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <mfem/fem/coefficient.hpp>
#include <mfem/fem/gridfunc.hpp>

using namespace std;
using namespace mfem;

// Custom integraition of boundary conditions function declaration
real_t IntegrateBC(const GridFunction &sol, const Array<int> &bdr_marker,
                   real_t alpha, real_t beta, real_t gamma, real_t &error);

int main(int argc, char *argv[]) {
  // Parameters for the analysis.
  const char *mesh_file = "2D_Grain.msh";
  int order = 1;
  // int order = 0;
  bool static_cond = false;
  bool pa = false;
  bool fa = false;
  bool h1 = true;
  const char *device_config = "cpu";
  bool visualization = true;
  bool algebraic_ceed = false;

  real_t mat_val = 1.30; // SiO_2 heat capacity[W/m-K]
  // real_t mat_val = 1000000.0; // SiO_2 heat conductivity coefficient
  // [J/m^3-K]
  real_t dbc_val = 0.0;
  real_t nbc_val = 500.0; // Heat flux assumed at 500 [W/m^2]
  real_t rbc_a_val = 0.0; // du/dn + a * u = b
  real_t rbc_b_val = 0.0;

  real_t kappa = -1.0;
  real_t sigma = -1.0;

  // Parse command-line options.
  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
  args.AddOption(&order, "-o", "--order",
                 "Finite element order (polynomial degree) or -1 for"
                 " isoparametric space.");
  args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                 "--no-static-condensation", "Enable static condensation.");
  args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                 "--no-partial-assembly", "Enable Partial Assembly.");
  args.AddOption(&fa, "-fa", "--full-assembly", "-no-fa", "--no-full-assembly",
                 "Enable Full Assembly.");
  args.AddOption(&device_config, "-d", "--device",
                 "Device configuration string, see Device::Configure().");
#ifdef MFEM_USE_CEED
  args.AddOption(&algebraic_ceed, "-a", "--algebraic", "-no-a",
                 "--no-algebraic", "Use algebraic Ceed solver");
#endif
  args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                 "--no-visualization",
                 "Enable or disable GLVis visualization.");
  args.Parse();
  if (!args.Good()) {
    args.PrintUsage(cout);
    return 1;
  }
  args.PrintOptions(cout);

  // Read the mesh from the given mesh file.
  Mesh mesh(mesh_file, 1, 1);
  int dim = mesh.Dimension();

  // Define a finite element space on the mesh. Here we use continuous
  // Lagrange finite elements of the specified order. If order < 1, we
  // instead use an isoparametric/isogeometric space.
  FiniteElementCollection *fec;
  bool delete_fec;
  if (order > 0) {
    fec = new H1_FECollection(order, dim);
    delete_fec = true;
  } else if (mesh.GetNodes()) {
    fec = mesh.GetNodes()->OwnFEC();
    delete_fec = false;
    cout << "Using isoparametric FEs: " << fec->Name() << endl;
  } else {
    fec = new H1_FECollection(order = 1, dim);
    delete_fec = true;
  }
  FiniteElementSpace fespace(&mesh, fec);
  cout << "Number of finite element unknowns: " << fespace.GetTrueVSize()
       << endl;

  // Create "marker arrays" to define the portions of boundary associated
  // with each type of boundary condition. These arrays have an entry
  // corresponding to each boundary attribute.  Placing a '1' in entry i
  // marks attribute i+1 as being active, '0' is inactive.
  Array<int> nbc_bdr(mesh.bdr_attributes.Max());
  Array<int> rbc_bdr(mesh.bdr_attributes.Max());
  Array<int> dbc_bdr(mesh.bdr_attributes.Max());

  nbc_bdr = 1;
  // nbc_bdr[0] = 1;
  cout << nbc_bdr.Size() << endl;
  for (int i = 0; i < nbc_bdr.Size(); i++) {
    nbc_bdr[i] = i;
    cout << nbc_bdr[i] << endl;
  }

  rbc_bdr = 0;
  rbc_bdr[1] = 0;
  dbc_bdr = 0;
  dbc_bdr[2] = 0;

  Array<int> ess_tdof_list;

  if (h1 && mesh.bdr_attributes.Size()) {
    cout << mesh.bdr_attributes.Max() << endl;
    Array<int> ess_tdof_list(mesh.bdr_attributes.Max());
    // For a continuous basis the linear system must be modified to enforce an
    // essential (Dirichlet) boundary condition. In the DG case this is not
    // necessary as the boundary condition will only be enforced weakly.
    fespace.GetEssentialTrueDofs(dbc_bdr, ess_tdof_list);
  }

  // 5. Setup the various coefficients needed for the Laplace operator and the
  //    various boundary conditions. In general these coefficients could be
  //    functions of position but here we use only constants.

  ConstantCoefficient matCoef(mat_val);
  ConstantCoefficient dbcCoef(dbc_val);
  ConstantCoefficient nbcCoef(nbc_val);
  ConstantCoefficient rbcACoef(rbc_a_val);
  ConstantCoefficient rbcBCoef(rbc_b_val);

  // Since the n.Grad(u) terms arise by integrating -Div(m Grad(u)) by parts we
  // must introduce the coefficient 'm' into the boundary conditions.
  // Therefore, in the case of the Neumann BC, we actually enforce m n.Grad(u)
  // = m g rather than simply n.Grad(u) = g.
  ProductCoefficient m_nbcCoef(matCoef, nbcCoef);
  ProductCoefficient m_rbcACoef(matCoef, rbcACoef);
  ProductCoefficient m_rbcBCoef(matCoef, rbcBCoef);

  // 6. Define the solution vector u as a finite element grid function
  //    corresponding to fespace. Initialize u with initial guess of zero.
  GridFunction u(&fespace);
  u = 0.0;

  // 7. Set up the bilinear form a(.,.) on the finite element space
  //    corresponding to the Laplacian operator -Delta, by adding the Diffusion
  //    domain integrator.
  BilinearForm a(&fespace);
  a.AddDomainIntegrator(new DiffusionIntegrator(matCoef));
  if (h1) {
    // Add a Mass integrator on the Robin boundary
    a.AddBoundaryIntegrator(new MassIntegrator(m_rbcACoef), rbc_bdr);
  } else {
    // Add the interfacial portion of the Laplace operator
    a.AddInteriorFaceIntegrator(
        new DGDiffusionIntegrator(matCoef, sigma, kappa));

    // Counteract the n.Grad(u) term on the Dirichlet portion of the boundary
    a.AddBdrFaceIntegrator(new DGDiffusionIntegrator(matCoef, sigma, kappa),
                           dbc_bdr);

    // Augment the n.Grad(u) term with a*u on the Robin portion of boundary
    a.AddBdrFaceIntegrator(new BoundaryMassIntegrator(m_rbcACoef), rbc_bdr);
  }
  a.Assemble();

  // 8. Assemble the linear form for the right hand side vector.
  LinearForm b(&fespace);

  if (h1) {
    // Set the Dirichlet values in the solution vector
    u.ProjectBdrCoefficient(dbcCoef, dbc_bdr);

    // Add the desired value for n.Grad(u) on the Neumann boundary
    b.AddBoundaryIntegrator(new BoundaryLFIntegrator(m_nbcCoef), nbc_bdr);

    // Add the desired value for n.Grad(u) + a*u on the Robin boundary
    b.AddBoundaryIntegrator(new BoundaryLFIntegrator(m_rbcBCoef), rbc_bdr);
  } else {
    // Add the desired value for the Dirichlet boundary
    b.AddBdrFaceIntegrator(
        new DGDirichletLFIntegrator(dbcCoef, matCoef, sigma, kappa), dbc_bdr);

    // Add the desired value for n.Grad(u) on the Neumann boundary
    b.AddBdrFaceIntegrator(new BoundaryLFIntegrator(m_nbcCoef), nbc_bdr);

    // Add the desired value for n.Grad(u) + a*u on the Robin boundary
    b.AddBdrFaceIntegrator(new BoundaryLFIntegrator(m_rbcBCoef), rbc_bdr);
  }
  b.Assemble();

  // 9. Construct the linear system.
  OperatorPtr A;
  Vector B, X;
  a.FormLinearSystem(ess_tdof_list, u, b, A, X, B);

#ifndef MFEM_USE_SUITESPARSE
  // 10. Define a simple symmetric Gauss-Seidel preconditioner and use it to
  //     solve the system AX=B with PCG in the symmetric case, and GMRES in the
  //     non-symmetric one.
  {
    GSSmoother M((SparseMatrix &)(*A));
    if (sigma == -1.0) {
      PCG(*A, M, B, X, 1, 500, 1e-12, 0.0);
    } else {
      GMRES(*A, M, B, X, 1, 500, 10, 1e-12, 0.0);
    }
  }
#else
  // 11. If MFEM was compiled with SuiteSparse, use UMFPACK to solve the
  //     system.
  UMFPackSolver umf_solver;
  umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
  umf_solver.SetOperator(*A);
  umf_solver.Mult(B, X);
#endif

  // 12. Recover the grid function corresponding to U. This is the local finite
  //     element solution.
  a.RecoverFEMSolution(X, b, u);

  // 13. Compute the various boundary integrals.
  mfem::out << endl
            << "Verifying boundary conditions" << endl
            << "=============================" << endl;
  {
    // Integrate the solution on the Dirichlet boundary and compare to the
    // expected value.
    real_t error, avg = IntegrateBC(u, dbc_bdr, 0.0, 1.0, dbc_val, error);

    bool hom_dbc = (dbc_val == 0.0);
    error /= hom_dbc ? 1.0 : fabs(dbc_val);
    mfem::out << "Average of solution on Gamma_dbc:\t" << avg << ", \t"
              << (hom_dbc ? "absolute" : "relative") << " error " << error
              << endl;
  }
  {
    // Integrate n.Grad(u) on the inhomogeneous Neumann boundary and compare
    // to the expected value.
    real_t error, avg = IntegrateBC(u, nbc_bdr, 1.0, 0.0, nbc_val, error);

    bool hom_nbc = (nbc_val == 0.0);
    error /= hom_nbc ? 1.0 : fabs(nbc_val);
    mfem::out << "Average of n.Grad(u) on Gamma_nbc:\t" << avg << ", \t"
              << (hom_nbc ? "absolute" : "relative") << " error " << error
              << endl;
  }
  {
    // Integrate n.Grad(u) on the homogeneous Neumann boundary and compare to
    // the expected value of zero.
    Array<int> nbc0_bdr(mesh.bdr_attributes.Max());
    nbc0_bdr = 0;
    nbc0_bdr[3] = 1;

    real_t error, avg = IntegrateBC(u, nbc0_bdr, 1.0, 0.0, 0.0, error);

    bool hom_nbc = true;
    mfem::out << "Average of n.Grad(u) on Gamma_nbc0:\t" << avg << ", \t"
              << (hom_nbc ? "absolute" : "relative") << " error " << error
              << endl;
  }
  {
    // Integrate n.Grad(u) + a * u on the Robin boundary and compare to the
    // expected value.
    real_t error;
    real_t avg = IntegrateBC(u, rbc_bdr, 1.0, rbc_a_val, rbc_b_val, error);

    bool hom_rbc = (rbc_b_val == 0.0);
    error /= hom_rbc ? 1.0 : fabs(rbc_b_val);
    mfem::out << "Average of n.Grad(u)+a*u on Gamma_rbc:\t" << avg << ", \t"
              << (hom_rbc ? "absolute" : "relative") << " error " << error
              << endl;
  }

  // 14. Save the refined mesh and the solution. This output can be viewed
  //     later using GLVis: "glvis -m refined.mesh -g sol.gf".
  {
    ofstream mesh_ofs("refined.mesh");
    mesh_ofs.precision(8);
    mesh.Print(mesh_ofs);
    ofstream sol_ofs("sol.gf");
    sol_ofs.precision(8);
    u.Save(sol_ofs);
  }

  if (visualization) {
    string title_str = h1 ? "H1" : "DG";
    char vishost[] = "localhost";
    int visport = 19916;
    socketstream sol_sock(vishost, visport);
    sol_sock.precision(8);
    sol_sock << "solution\n"
             << mesh << u << "window_title '" << title_str << " Solution'"
             << " keys 'mmc'" << flush;
  }

  // 15. Free the used memory.
  if (delete_fec) {
    delete fec;
  }

  return 0;
}

// Custom integration function for boundary conditons defintion

real_t IntegrateBC(const GridFunction &x, const Array<int> &bdr, real_t alpha,
                   real_t beta, real_t gamma, real_t &error) {
  real_t nrm = 0.0;
  real_t avg = 0.0;
  error = 0.0;

  const bool a_is_zero = alpha == 0.0;
  const bool b_is_zero = beta == 0.0;

  const FiniteElementSpace &fes = *x.FESpace();
  MFEM_ASSERT(fes.GetVDim() == 1, "");
  Mesh &mesh = *fes.GetMesh();
  Vector shape, loc_dofs, w_nor;
  DenseMatrix dshape;
  Array<int> dof_ids;
  for (int i = 0; i < mesh.GetNBE(); i++) {
    if (bdr[mesh.GetBdrAttribute(i) - 1] == 0) {
      continue;
    }

    FaceElementTransformations *FTr = mesh.GetBdrFaceTransformations(i);
    if (FTr == nullptr) {
      continue;
    }

    const FiniteElement &fe = *fes.GetFE(FTr->Elem1No);
    MFEM_ASSERT(fe.GetMapType() == FiniteElement::VALUE, "");
    const int int_order = 2 * fe.GetOrder() + 3;
    const IntegrationRule &ir = IntRules.Get(FTr->FaceGeom, int_order);

    fes.GetElementDofs(FTr->Elem1No, dof_ids);
    x.GetSubVector(dof_ids, loc_dofs);
    if (!a_is_zero) {
      const int sdim = FTr->Face->GetSpaceDim();
      w_nor.SetSize(sdim);
      dshape.SetSize(fe.GetDof(), sdim);
    }
    if (!b_is_zero) {
      shape.SetSize(fe.GetDof());
    }
    for (int j = 0; j < ir.GetNPoints(); j++) {
      const IntegrationPoint &ip = ir.IntPoint(j);
      IntegrationPoint eip;
      FTr->Loc1.Transform(ip, eip);
      FTr->Face->SetIntPoint(&ip);
      real_t face_weight = FTr->Face->Weight();
      real_t val = 0.0;
      if (!a_is_zero) {
        FTr->Elem1->SetIntPoint(&eip);
        fe.CalcPhysDShape(*FTr->Elem1, dshape);
        CalcOrtho(FTr->Face->Jacobian(), w_nor);
        val += alpha * dshape.InnerProduct(w_nor, loc_dofs) / face_weight;
      }
      if (!b_is_zero) {
        fe.CalcShape(eip, shape);
        val += beta * (shape * loc_dofs);
      }

      // Measure the length of the boundary
      nrm += ip.weight * face_weight;

      // Integrate alpha * n.Grad(x) + beta * x
      avg += val * ip.weight * face_weight;

      // Integrate |alpha * n.Grad(x) + beta * x - gamma|^2
      val -= gamma;
      error += (val * val) * ip.weight * face_weight;
    }
  }

  // Normalize by the length of the boundary
  if (std::abs(nrm) > 0.0) {
    error /= nrm;
    avg /= nrm;
  }

  // Compute l2 norm of the error in the boundary condition (negative
  // quadrature weights may produce negative 'error')
  error = (error >= 0.0) ? sqrt(error) : -sqrt(-error);

  // Return the average value of alpha * n.Grad(x) + beta * x
  return avg;
}
